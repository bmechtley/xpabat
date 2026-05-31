// ─── WebGL2 tile renderer ──────────────────────────────────────────────────────
// Uses hardware trilinear mipmaps (LINEAR_MIPMAP_LINEAR) for alias-free rendering
// at extreme zoom-out ratios where Canvas 2D drawImage aliases even at 'high'
// quality. Map renderers, games, and any proper GPU renderer use this approach.
//
// Public API (called from render.js):
//   _glInit(W, H)        – ensure the offscreen GL canvas is ready; returns bool
//   _glDrawTile(...)     – draw one tile quad into the GL canvas
//   clearGLTextures()    – evict all cached textures (call with warp cache clears)
//
// Usage pattern in render.js:
//   1. if (S.useWebGL && _glInit(W, H)) { _gl.clear(...); }
//   2. For each tile: _glDrawTile(...)
//   3. After tile loop: ctx.drawImage(_glCanvas, 0, 0)  — blit onto main canvas

let _gl        = null;
let _glCanvas  = null;
let _glProg    = null;
let _glLoc     = {};             // cached uniform / attrib locations
let _glVbo     = null;
const _glQuad  = new Float32Array(24);  // scratch buffer: 6 verts × (x,y,u,v)
const _glTextures = new Map();   // texKey → WebGLTexture

// ── Initialise (or resize) the offscreen WebGL2 canvas ──────────────────────
// Returns true on success, false if WebGL2 is unavailable / shader compile failed.
function _glInit(W, H) {
  // No-op if already the right size
  if (_glCanvas && _glCanvas.width === W && _glCanvas.height === H) return true;

  if (!_glCanvas) {
    const cv = document.createElement('canvas');
    const gl = cv.getContext('webgl2', {
      alpha: true,             // GL canvas composites over the 2D background
      premultipliedAlpha: false,
      antialias: false,
      depth: false,
      stencil: false,
    });
    if (!gl) { console.warn('[xpabat] WebGL2 not available — using Canvas 2D path'); return false; }

    _glCanvas = cv;
    _gl = gl;

    if (!_glBuildProgram()) {
      _glCanvas = null;
      _gl = null;
      return false;
    }

    _gl.enable(_gl.BLEND);
    _gl.blendFunc(_gl.SRC_ALPHA, _gl.ONE_MINUS_SRC_ALPHA);
    _glVbo = _gl.createBuffer();
    _gl.activeTexture(_gl.TEXTURE0);
    _gl.uniform1i(_glLoc.tex, 0);
  }

  _glCanvas.width  = W;
  _glCanvas.height = H;
  _gl.viewport(0, 0, W, H);
  _gl.uniform2f(_glLoc.res, W, H);
  return true;
}

function _glBuildProgram() {
  // Vertex shader: pixel-space → NDC (y-flipped to match canvas convention)
  const vsrc = `#version 300 es
    in  vec2 a_pos;
    in  vec2 a_uv;
    uniform vec2 u_res;
    out vec2 v_uv;
    void main() {
      vec2 ndc = a_pos / u_res * 2.0 - 1.0;
      ndc.y    = -ndc.y;
      gl_Position = vec4(ndc, 0.0, 1.0);
      v_uv = a_uv;
    }`;

  // Fragment shader: trilinear sample + luminance-preserving saturation
  const fsrc = `#version 300 es
    precision mediump float;
    in  vec2 v_uv;
    uniform sampler2D u_tex;
    uniform float     u_alpha;
    uniform float     u_sat;
    out vec4 fragColor;
    void main() {
      vec4 c = texture(u_tex, v_uv);
      // Luminance-preserving saturation — matches CSS saturate() coefficient
      float lum = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      c.rgb = mix(vec3(lum), c.rgb, u_sat);
      fragColor = vec4(c.rgb, c.a * u_alpha);
    }`;

  const compile = (type, src) => {
    const s = _gl.createShader(type);
    _gl.shaderSource(s, src);
    _gl.compileShader(s);
    if (!_gl.getShaderParameter(s, _gl.COMPILE_STATUS)) {
      console.error('[xpabat WebGL] shader compile error:', _gl.getShaderInfoLog(s));
      _gl.deleteShader(s);
      return null;
    }
    return s;
  };

  const vs = compile(_gl.VERTEX_SHADER,   vsrc);
  const fs = compile(_gl.FRAGMENT_SHADER, fsrc);
  if (!vs || !fs) return false;

  _glProg = _gl.createProgram();
  _gl.attachShader(_glProg, vs);
  _gl.attachShader(_glProg, fs);
  _gl.linkProgram(_glProg);
  _gl.deleteShader(vs);
  _gl.deleteShader(fs);

  if (!_gl.getProgramParameter(_glProg, _gl.LINK_STATUS)) {
    console.error('[xpabat WebGL] link error:', _gl.getProgramInfoLog(_glProg));
    return false;
  }

  _gl.useProgram(_glProg);
  _glLoc.aPos  = _gl.getAttribLocation (_glProg, 'a_pos');
  _glLoc.aUV   = _gl.getAttribLocation (_glProg, 'a_uv');
  _glLoc.res   = _gl.getUniformLocation(_glProg, 'u_res');
  _glLoc.tex   = _gl.getUniformLocation(_glProg, 'u_tex');
  _glLoc.alpha = _gl.getUniformLocation(_glProg, 'u_alpha');
  _glLoc.sat   = _gl.getUniformLocation(_glProg, 'u_sat');
  return true;
}

// ── Upload a canvas as a GL texture (cached, no mipmaps) ────────────────────
// The caller is responsible for pre-downsampling to ≤ 2:1 before uploading
// (via _getWarpedTileBlit), so GPU mipmaps are unnecessary and omitted.
// Bilinear (LINEAR) min+mag filter gives clean results at ≤ 2:1 ratios.
function _glGetTex(srcCanvas, key) {
  if (_glTextures.has(key)) return _glTextures.get(key);

  const tex = _gl.createTexture();
  _gl.bindTexture(_gl.TEXTURE_2D, tex);
  _gl.texImage2D(_gl.TEXTURE_2D, 0, _gl.RGBA, _gl.RGBA, _gl.UNSIGNED_BYTE, srcCanvas);
  _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_MIN_FILTER, _gl.LINEAR);
  _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_MAG_FILTER, _gl.LINEAR);
  _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_WRAP_S,     _gl.CLAMP_TO_EDGE);
  _gl.texParameteri(_gl.TEXTURE_2D, _gl.TEXTURE_WRAP_T,     _gl.CLAMP_TO_EDGE);
  _glTextures.set(key, tex);
  return tex;
}

// ── Evict all cached GL textures ────────────────────────────────────────────
// Call this wherever S.tileWarpCache.clear() is called (resize, logScale change).
// The texture keys mirror the warp-canvas keys, so both caches stay in sync.
function clearGLTextures() {
  if (!_gl) return;
  for (const t of _glTextures.values()) _gl.deleteTexture(t);
  _glTextures.clear();
}


// ── Draw one tile quad into _glCanvas ───────────────────────────────────────
// Mirrors the Canvas 2D tile blit but with hardware trilinear mipmaps.
//
// idx, img, H       – tile index, Image element, canvas height in px
// srcX0, srcW       – source X slice in the warp canvas (time axis, pixels)
// dstX0, dstW       – destination X range on the GL canvas (pixels)
// alpha             – 1.0 for main tiles; S.flatness / S.crossfade for overlays
// sat               – S.saturation (1 = full colour, 0 = greyscale)
// warpCache         – S.tileWarpCache, S.flatTileWarpCache, or S.maskTileWarpCache
//
// Returns false when the log-warp canvas isn't ready yet (budget exceeded);
// the caller should render a linear fallback and schedule another frame.
function _glDrawTile(idx, img, H, srcX0, srcW, dstX0, dstW, alpha, sat, warpCache) {
  // Use the same pre-downsampled canvas as the Canvas 2D path: _getWarpedTileBlit
  // halves the warp canvas until srcW/dstW ≤ 2:1 using the browser's high-quality
  // imageSmoothingQuality='high' filter.  Uploading this pre-downsampled canvas
  // (rather than the full-res warp + GPU generateMipmap box filter) gives the same
  // clean area-averaged look as Canvas 2D at all zoom levels.
  const blit = _getWarpedTileBlit(idx, img, H, srcX0, srcW, dstW, warpCache);
  if (!blit) return false;   // log warp budget exceeded this frame

  // Texture key: warp-canvas key + blit canvas width so each downsampling level
  // gets its own cached GL texture.  clearGLTextures() evicts all on cache clear.
  const pfx = (warpCache === S.maskTileWarpCache) ? 'k'
             : (warpCache === S.flatTileWarpCache) ? 'f' : 'm';
  const texKey = `${pfx}-${idx}-${H}-${S.logScale.toFixed(3)}-${blit.canvas.width}`;
  const tex    = _glGetTex(blit.canvas, texKey);

  // U: source X within the blit canvas (coordinates already scaled by _getWarpedTileBlit)
  const tW = blit.canvas.width;
  const u0 = blit.sx / tW;
  const u1 = (blit.sx + blit.sw) / tW;

  // V: current freq viewport mapped onto the full-range warp canvas Y axis.
  // _fullRangeFToY gives the Y pixel (0 = TILE_FREQ_HIGH, H = TILE_FREQ_LOW).
  const wY0 = _fullRangeFToY(S.freqHigh, H, S.logScale);
  const wY1 = _fullRangeFToY(S.freqLow,  H, S.logScale);
  if (wY1 <= wY0) return true;
  const v0 = wY0 / H;
  const v1 = wY1 / H;

  // Destination rect (screen pixels): full canvas height, subset of width
  const dx0 = dstX0;
  const dx1 = dstX0 + dstW;
  const dy0 = 0;
  const dy1 = H;

  // Two triangles forming a quad: TL TR BL | TR BR BL
  const d = _glQuad;
  d[ 0]=dx0; d[ 1]=dy0; d[ 2]=u0; d[ 3]=v0;   // TL
  d[ 4]=dx1; d[ 5]=dy0; d[ 6]=u1; d[ 7]=v0;   // TR
  d[ 8]=dx0; d[ 9]=dy1; d[10]=u0; d[11]=v1;   // BL
  d[12]=dx1; d[13]=dy0; d[14]=u1; d[15]=v0;   // TR
  d[16]=dx1; d[17]=dy1; d[18]=u1; d[19]=v1;   // BR
  d[20]=dx0; d[21]=dy1; d[22]=u0; d[23]=v1;   // BL

  const gl = _gl;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.bindBuffer(gl.ARRAY_BUFFER, _glVbo);
  gl.bufferData(gl.ARRAY_BUFFER, d, gl.DYNAMIC_DRAW);
  // stride = 4 floats × 4 bytes = 16; a_pos at offset 0, a_uv at offset 8
  gl.vertexAttribPointer(_glLoc.aPos, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(_glLoc.aPos);
  gl.vertexAttribPointer(_glLoc.aUV,  2, gl.FLOAT, false, 16, 8);
  gl.enableVertexAttribArray(_glLoc.aUV);
  gl.uniform1f(_glLoc.alpha, alpha);
  gl.uniform1f(_glLoc.sat,   sat);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  return true;
}
