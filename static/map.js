// ─── File map picker ──────────────────────────────────────────────────────────
// Shows all recordings as pins on a Leaflet map.  Clicking a pin opens a popup
// listing the files at that location with a "Select" button each.

const FILE_LOCATIONS = [
  {
    id: 'bracken',
    pattern: /bracken|san.antonio/i,
    lat: 29.7061, lng: -98.4153,
    name: 'Bracken Bat Cave',
    desc: 'Bracken Bat Cave, Medina County\nNear San Antonio, TX',
  },
  {
    id: 'campbell',
    pattern: /.*/,          // default — catches everything else
    lat: 32.28408, lng: -110.94397,
    name: 'Campbell Ave Bridge',
    desc: 'Campbell Ave Bridge over Rillito River\nTucson, AZ',
  },
];

let _mapInst   = null;   // Leaflet map instance (created once)
let _mapMarkers = [];    // active marker objects

function _fileLocation(name) {
  for (const loc of FILE_LOCATIONS) {
    if (loc.pattern.test(name)) return loc;
  }
  return null;
}

function _prettyDate(name) {
  // e.g. "2025-05-28 1942" → "2025-05-28 19:42"
  //      "2025-09-20-1732"  → "2025-09-20 17:32"
  const m = name.match(/(\d{4}[-_ ]\d{2}[-_ ]\d{2})[- _]?(\d{4})/);
  if (!m) return '';
  const d = m[1].replace(/[_ ]/g, '-');
  const t = m[2].slice(0, 2) + ':' + m[2].slice(2);
  return d + '  ' + t;
}

async function openFileMap() {
  const modal = document.getElementById('map-modal');
  modal.classList.add('open');

  // Fetch file list
  let files;
  try {
    const j = await (await fetch(`/api/files?f=${S.fid}`)).json();
    files = j.files;
  } catch { return; }

  // Initialise Leaflet on first open
  if (!_mapInst) {
    _mapInst = L.map('map-leaflet', { zoomControl: true });
    L.tileLayer('/api/maptile/{s}/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> © <a href="https://carto.com/attributions" target="_blank">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(_mapInst);
  }

  // Clear old markers
  for (const m of _mapMarkers) m.remove();
  _mapMarkers = [];

  // Group files by location
  const byLoc = {};
  for (const f of files) {
    const loc = _fileLocation(f.name);
    if (!loc) continue;
    if (!byLoc[loc.id]) byLoc[loc.id] = { loc, files: [] };
    byLoc[loc.id].files.push(f);
  }

  const points = [];

  for (const { loc, files: locFiles } of Object.values(byLoc)) {
    points.push([loc.lat, loc.lng]);
    const isCurrent = locFiles.some(f => f.fid === S.fid);

    const icon = L.divIcon({
      className: '',
      html: `<div class="map-pin${isCurrent ? ' map-pin-active' : ''}"></div>`,
      iconSize: [20, 20],
      iconAnchor: [10, 10],
      popupAnchor: [0, -12],
    });

    const filesHtml = locFiles.map((f, i) => {
      const current = f.fid === S.fid;
      return `
        <div class="mp-file${current ? ' mp-file-current' : ''}">
          <div class="mp-fname">${f.name}</div>
          <div class="mp-fdate">${_prettyDate(f.name)}</div>
          ${current
            ? '<span class="mp-cur-badge">Currently open</span>'
            : `<button class="mp-sel-btn" onclick="closeFileMap();switchFile('${f.fid}')">Select</button>`}
        </div>
        ${i < locFiles.length - 1 ? '<div class="mp-sep"></div>' : ''}`;
    }).join('');

    const popup = L.popup({ maxWidth: 300, className: 'map-popup-wrap' }).setContent(`
      <div class="mp-popup">
        <div class="mp-loc-name">${loc.name}</div>
        <div class="mp-loc-desc">${loc.desc.replace('\n', '<br>')}</div>
        ${filesHtml}
      </div>`);

    const marker = L.marker([loc.lat, loc.lng], { icon }).bindPopup(popup);
    if (isCurrent) marker.on('add', () => marker.openPopup());
    marker.addTo(_mapInst);
    _mapMarkers.push(marker);
  }

  // Fit map to all markers, then open current-file popup
  if (points.length > 1) {
    _mapInst.fitBounds(points, { padding: [60, 60] });
  } else if (points.length === 1) {
    _mapInst.setView(points[0], 14);
  }

  // Leaflet needs the container to be visible before sizing
  setTimeout(() => {
    _mapInst.invalidateSize();
    // Open popup for the current file's location
    for (const m of _mapMarkers) {
      const el = m.getElement();
      if (el && el.querySelector('.map-pin-active')) m.openPopup();
    }
  }, 80);
}

function closeFileMap() {
  document.getElementById('map-modal').classList.remove('open');
}
