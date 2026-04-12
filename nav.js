/**
 * nav.js — GPS navigation utilities
 * Converts drone GPS + firefighter GPS into actionable routing output.
 */

const R = 6371000; // Earth radius in metres

export function toRad(deg) { return deg * Math.PI / 180; }
export function toDeg(rad) { return rad * 180 / Math.PI; }

/**
 * Haversine distance between two GPS points, in metres.
 */
export function haversineDistance(lat1, lon1, lat2, lon2) {
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 +
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Bearing in degrees (0 = North, 90 = East) from point A to point B.
 */
export function bearing(lat1, lon1, lat2, lon2) {
  const dLon = toRad(lon2 - lon1);
  const y = Math.sin(dLon) * Math.cos(toRad(lat2));
  const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
             Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLon);
  return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

/**
 * Convert bearing degrees to compass rose label.
 */
export function bearingLabel(deg) {
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return dirs[Math.round(deg / 22.5) % 16];
}

/**
 * Estimate altitude context from drone altitude above takeoff point.
 * DJI Mini 2 SE reports altitude in metres AGL (above ground level at takeoff).
 */
export function altitudeContext(altMetres) {
  if (altMetres === null || altMetres === undefined) return 'unknown altitude';
  if (altMetres < 4)  return `ground level (~${altMetres.toFixed(1)}m)`;
  if (altMetres < 8)  return `~1st–2nd floor (~${altMetres.toFixed(1)}m)`;
  if (altMetres < 14) return `~2nd–4th floor (~${altMetres.toFixed(1)}m)`;
  if (altMetres < 20) return `~4th–6th floor (~${altMetres.toFixed(1)}m)`;
  return `high elevation (~${altMetres.toFixed(1)}m)`;
}

/**
 * Full navigation briefing from firefighter position to target GPS.
 * @param {number} ffLat  - firefighter latitude
 * @param {number} ffLon  - firefighter longitude
 * @param {object} gps    - {lat, lon, alt, heading} of drone at detection time
 * @returns {object}      - navigation object ready to render
 */
export function buildNavigation(ffLat, ffLon, gps) {
  if (!gps || gps.lat == null || gps.lon == null) {
    return { error: 'No GPS data for this detection.' };
  }

  const dist = haversineDistance(ffLat, ffLon, gps.lat, gps.lon);
  const brng = bearing(ffLat, ffLon, gps.lat, gps.lon);
  const label = bearingLabel(brng);
  const alt = altitudeContext(gps.alt);

  return {
    distance: Math.round(dist),
    bearing: Math.round(brng),
    bearingLabel: label,
    altContext: alt,
    droneAlt: gps.alt,
    targetLat: gps.lat,
    targetLon: gps.lon,
    mapsUrl: `https://www.google.com/maps/dir/${ffLat},${ffLon}/${gps.lat},${gps.lon}`,
    satelliteUrl: `https://www.google.com/maps/@${gps.lat},${gps.lon},50m/data=!3m1!1e3`,
    summary: `${Math.round(dist)}m ${label} (${Math.round(brng)}°) — ${alt}`,
  };
}

/**
 * Draw a compass rose SVG string for a given bearing.
 */
export function compassSVG(bearingDeg, size = 80) {
  const cx = size / 2, cy = size / 2, r = size / 2 - 4;
  // Arrow tip points in bearing direction from North (up = 0°)
  const rad = toRad(bearingDeg - 90); // SVG 0° is right; offset -90 for North-up
  const arrowX = cx + r * 0.6 * Math.cos(rad);
  const arrowY = cy + r * 0.6 * Math.sin(rad);
  const tailX  = cx - r * 0.4 * Math.cos(rad);
  const tailY  = cy - r * 0.4 * Math.sin(rad);

  return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,140,0,0.25)" stroke-width="1"/>
    <text x="${cx}" y="10" text-anchor="middle" font-size="8" fill="rgba(255,140,0,0.5)" font-family="monospace">N</text>
    <text x="${cx}" y="${size-2}" text-anchor="middle" font-size="8" fill="rgba(255,140,0,0.3)" font-family="monospace">S</text>
    <text x="6" y="${cy+3}" text-anchor="middle" font-size="8" fill="rgba(255,140,0,0.3)" font-family="monospace">W</text>
    <text x="${size-6}" y="${cy+3}" text-anchor="middle" font-size="8" fill="rgba(255,140,0,0.3)" font-family="monospace">E</text>
    <line x1="${tailX}" y1="${tailY}" x2="${arrowX}" y2="${arrowY}"
      stroke="#ff8c00" stroke-width="2.5" stroke-linecap="round"
      marker-end="url(#nav-arrow)"/>
    <defs>
      <marker id="nav-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M0,1 L8,5 L0,9 Z" fill="#ff8c00"/>
      </marker>
    </defs>
  </svg>`;
}
