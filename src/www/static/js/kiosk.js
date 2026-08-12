/* TARS Kiosk Display — SocketIO client */

const socket = io();

const $messages   = document.getElementById('messages');
const $status     = document.getElementById('status-badge');
const $statusText = document.getElementById('status-text');
const $silenceFill = document.getElementById('silence-fill');
const $toolbarStatus = document.getElementById('toolbar-status');
const $toolbarTime = document.getElementById('toolbar-time');
const $battery    = document.getElementById('battery');
const $cpuTemp    = document.getElementById('cpu-temp');
const $dnd        = document.getElementById('dnd-indicator');
const $screensaver = document.getElementById('screensaver');
const $screensaverCanvas = document.getElementById('screensaver-canvas');
const $screensaverClock = document.getElementById('screensaver-clock');
const $overlayWrap = document.getElementById('overlay-image');
const $overlayImg  = document.getElementById('overlay-img');
const $powerMenu  = document.getElementById('power-menu');
const $toolbarInfo = document.getElementById('toolbar-info');
const $wifi       = document.getElementById('wifi-indicator');

const $micCanvas  = document.getElementById('mic-level');
const micCtx      = $micCanvas.getContext('2d');

let outdoorTemp = null;

let currentStatus = 'BOOTING';
let screensaverActive = false;
let screensaverTimer = null;
let lastStreamMsg = null;   // reference to the last bot message element (for streaming)
const MAX_MESSAGES = 50;

// Mic level state
let micSilenceProgress = 0;
let micSilenceMax = 1;
let micLevel = 0;        // actual RMS level 0.0-1.0 from backend
let micAnimFrame = null;

// ── Mic level indicator ────────────────────────────────────────────────────
function resizeMicCanvas() {
  $micCanvas.width = $micCanvas.parentElement.clientWidth || $micCanvas.offsetWidth;
}
resizeMicCanvas();
window.addEventListener('resize', resizeMicCanvas);

function drawMicLevel() {
  const w = $micCanvas.width;
  const h = $micCanvas.height;
  micCtx.clearRect(0, 0, w, h);

  const barCount = 40;
  const barWidth = w / barCount;
  const isListening = currentStatus === 'LISTENING';
  const isTalking = currentStatus === 'TALKING';
  const isStandby = currentStatus === 'STANDBY';

  if (!isListening && !isTalking && !isStandby) {
    micAnimFrame = null;
    return;
  }

  for (let i = 0; i < barCount; i++) {
    // Base level from actual mic RMS + per-bar variation for visual spread
    const variation = 0.7 + 0.6 * Math.sin(i * 0.8 + Date.now() * 0.003);
    let barLevel;

    if (isTalking) {
      barLevel = 0.3 + 0.7 * Math.abs(Math.sin((Date.now() / 200) + i * 0.3));
    } else {
      barLevel = Math.min(1.0, micLevel * variation);
    }

    const barH = Math.max(1, barLevel * h);
    const y = (h - barH) / 2;

    let color;
    if (isTalking) {
      color = `rgba(0, 255, 100, 0.7)`;
    } else if (isStandby) {
      color = `rgba(0, 120, 120, ${0.3 + micLevel * 0.6})`;
    } else {
      // LISTENING — cyan, brighter with louder input
      color = `rgba(0, 255, 255, ${0.3 + micLevel * 0.6})`;
    }

    micCtx.fillStyle = color;
    micCtx.fillRect(i * barWidth + 1, y, barWidth - 2, barH);
  }

  micAnimFrame = setTimeout(drawMicLevel, 66);  // ~15fps
}

function startMicAnimation() {
  if (!micAnimFrame) {
    micAnimFrame = setTimeout(drawMicLevel, 66);
  }
}

function stopMicAnimation() {
  if (micAnimFrame) {
    clearTimeout(micAnimFrame);
    micAnimFrame = null;
  }
  micCtx.clearRect(0, 0, $micCanvas.width, $micCanvas.height);
}

// ── Clock + outdoor temperature ────────────────────────────────────────────
function formatTime() {
  const now = new Date();
  const h = now.getHours();
  const m = String(now.getMinutes()).padStart(2, '0');
  if (AMPM) {
    return `${h % 12 || 12}:${m} ${h >= 12 ? 'PM' : 'AM'}`;
  }
  return `${String(h).padStart(2, '0')}:${m}`;
}

function updateClock() {
  const time = formatTime();
  const tempStr = outdoorTemp !== null ? ` | ${Math.round(outdoorTemp)}C` : '';
  $toolbarInfo.textContent = time + tempStr;

  if (screensaverActive && SHOW_TIME) {
    $screensaverClock.textContent = time + tempStr;
  }
}
setInterval(updateClock, 10000);
updateClock();

// Fetch outdoor temp from Open-Meteo (free, no API key)
function fetchOutdoorTemp() {
  if (!LATITUDE || !LONGITUDE) return;
  fetch(`https://api.open-meteo.com/v1/forecast?latitude=${LATITUDE}&longitude=${LONGITUDE}&current=temperature_2m&timezone=auto`)
    .then(r => r.json())
    .then(d => { outdoorTemp = d.current.temperature_2m; updateClock(); })
    .catch(() => {});
}
fetchOutdoorTemp();
setInterval(fetchOutdoorTemp, 600000); // refresh every 10 min

// ── Screensaver system ─────────────────────────────────────────────────────
const AVAILABLE_SCREENSAVERS = ['starfield', 'matrix', 'hyperspace', 'blackhole'];
const enabledScreensavers = SCREENSAVER_LIST.filter(s => AVAILABLE_SCREENSAVERS.includes(s));
if (enabledScreensavers.length === 0) enabledScreensavers.push('starfield');

let ssCtx, ssAnim = null, ssCycleTimer = null, ssCurrentIdx = 0;

function ssResize() {
  // Use parent dimensions (rotated container) not window
  const parent = $screensaverCanvas.parentElement;
  $screensaverCanvas.width = parent.clientWidth || window.innerHeight;
  $screensaverCanvas.height = parent.clientHeight || window.innerWidth;
  ssCtx = $screensaverCanvas.getContext('2d');
}

// ── Starfield ──
function initStarfield(w, h) {
  const stars = [];
  for (let i = 0; i < 150; i++)
    stars.push({ x: Math.random()*w, y: Math.random()*h, size: Math.random()*2+0.5, speed: Math.random()*0.3+0.1, alpha: Math.random() });
  return function draw() {
    ssCtx.fillStyle = '#000'; ssCtx.fillRect(0,0,w,h);
    for (const s of stars) {
      s.alpha += (Math.random()-0.5)*0.05; s.alpha = Math.max(0.2, Math.min(1, s.alpha));
      s.y += s.speed; if (s.y > h) { s.y = 0; s.x = Math.random()*w; }
      ssCtx.fillStyle = `rgba(0,255,255,${s.alpha})`; ssCtx.beginPath(); ssCtx.arc(s.x, s.y, s.size, 0, Math.PI*2); ssCtx.fill();
    }
  };
}

// ── Matrix ──
function initMatrix(w, h) {
  const fontSize = 14, cols = Math.floor(w/fontSize);
  const drops = Array(cols).fill(1);
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*(){}[]<>';
  return function draw() {
    ssCtx.fillStyle = 'rgba(0,0,0,0.05)'; ssCtx.fillRect(0,0,w,h);
    ssCtx.fillStyle = '#0f0'; ssCtx.font = fontSize + 'px monospace';
    for (let i = 0; i < drops.length; i++) {
      const ch = chars[Math.floor(Math.random()*chars.length)];
      const x = i * fontSize, y = drops[i] * fontSize;
      ssCtx.fillStyle = y < h*0.3 ? '#0f0' : `rgba(0,255,0,${0.8-y/h*0.6})`;
      ssCtx.fillText(ch, x, y);
      if (y > h && Math.random() > 0.975) drops[i] = 0;
      drops[i]++;
    }
  };
}

// ── Hyperspace ──
function initHyperspace(w, h) {
  const cx = w/2, cy = h/2;
  const stars = [];
  for (let i = 0; i < 200; i++)
    stars.push({ x: (Math.random()-0.5)*w*2, y: (Math.random()-0.5)*h*2, z: Math.random()*w });
  return function draw() {
    ssCtx.fillStyle = 'rgba(0,0,0,0.15)'; ssCtx.fillRect(0,0,w,h);
    for (const s of stars) {
      s.z -= 8;
      if (s.z <= 0) { s.x = (Math.random()-0.5)*w*2; s.y = (Math.random()-0.5)*h*2; s.z = w; }
      const sx = (s.x/s.z)*w*0.5 + cx, sy = (s.y/s.z)*h*0.5 + cy;
      const r = Math.max(0.5, (1-s.z/w)*3);
      const a = Math.min(1, (1-s.z/w)*1.5);
      ssCtx.fillStyle = `rgba(200,220,255,${a})`;
      // Draw streak
      const px = (s.x/(s.z+8))*w*0.5 + cx, py = (s.y/(s.z+8))*h*0.5 + cy;
      ssCtx.beginPath(); ssCtx.moveTo(px,py); ssCtx.lineTo(sx,sy); ssCtx.lineWidth = r;
      ssCtx.strokeStyle = `rgba(200,220,255,${a})`; ssCtx.stroke();
    }
  };
}

// ── Blackhole ──
function initBlackhole(w, h) {
  const cx = w/2, cy = h/2;
  const particles = [];
  for (let i = 0; i < 120; i++) {
    const angle = Math.random() * Math.PI * 2;
    const dist = 50 + Math.random() * Math.max(w,h) * 0.4;
    particles.push({ angle, dist, speed: 0.005 + Math.random()*0.015, size: Math.random()*2+0.5, decay: 0.998 + Math.random()*0.001 });
  }
  return function draw() {
    ssCtx.fillStyle = 'rgba(0,0,0,0.08)'; ssCtx.fillRect(0,0,w,h);
    // Glow at center
    const grd = ssCtx.createRadialGradient(cx,cy,2,cx,cy,30);
    grd.addColorStop(0, 'rgba(80,0,120,0.3)'); grd.addColorStop(1, 'rgba(0,0,0,0)');
    ssCtx.fillStyle = grd; ssCtx.fillRect(cx-30,cy-30,60,60);
    for (const p of particles) {
      p.angle += p.speed;
      p.dist *= p.decay;
      if (p.dist < 5) { p.dist = 50 + Math.random() * Math.max(w,h)*0.4; p.speed = 0.005 + Math.random()*0.015; }
      p.speed += 0.0001; // accelerate as it spirals in
      const x = cx + Math.cos(p.angle) * p.dist;
      const y = cy + Math.sin(p.angle) * p.dist;
      const a = Math.min(1, p.dist / 100);
      const hue = (p.dist < 80) ? '180,0,255' : '0,200,255';
      ssCtx.fillStyle = `rgba(${hue},${a})`;
      ssCtx.beginPath(); ssCtx.arc(x, y, p.size, 0, Math.PI*2); ssCtx.fill();
    }
  };
}

// ── Screensaver lifecycle ──
const ssFactories = { starfield: initStarfield, matrix: initMatrix, hyperspace: initHyperspace, blackhole: initBlackhole };

function startScreensaver(name) {
  ssResize();
  const w = $screensaverCanvas.width, h = $screensaverCanvas.height;
  const factory = ssFactories[name] || ssFactories.starfield;
  const drawFn = factory(w, h);
  if (ssAnim) clearInterval(ssAnim);
  ssAnim = setInterval(drawFn, 50); // ~20fps
}

function cycleScreensaver() {
  ssCurrentIdx = (ssCurrentIdx + 1) % enabledScreensavers.length;
  startScreensaver(enabledScreensavers[ssCurrentIdx]);
}

function activateScreensaver() {
  if (screensaverActive) return;
  screensaverActive = true;
  $screensaver.classList.remove('hidden');
  ssCurrentIdx = Math.floor(Math.random() * enabledScreensavers.length);
  startScreensaver(enabledScreensavers[ssCurrentIdx]);
  ssCycleTimer = setInterval(cycleScreensaver, SCREENSAVER_CYCLE_SEC * 1000);
  updateClock();
}

function deactivateScreensaver() {
  screensaverActive = false;
  $screensaver.classList.add('hidden');
  if (ssAnim) { clearInterval(ssAnim); ssAnim = null; }
  if (ssCycleTimer) { clearInterval(ssCycleTimer); ssCycleTimer = null; }
  resetScreensaverTimer();
}

function resetScreensaverTimer() {
  clearTimeout(screensaverTimer);
  screensaverTimer = setTimeout(activateScreensaver, SCREENSAVER_TIMEOUT_SEC * 1000);
}
resetScreensaverTimer();

// Touch/click dismisses screensaver
document.addEventListener('click', () => {
  if (screensaverActive) deactivateScreensaver();
});

// ── Messages ───────────────────────────────────────────────────────────────
function classifyMessage(source) {
  const upper = source.toUpperCase();
  if (upper === CHAR_NAME.toUpperCase()) return 'bot';
  if (['SYSTEM', 'SYS', 'INFO', 'WARNING', 'ERROR', 'DEBUG', 'LOAD'].includes(upper)) return 'system';
  return 'user';
}

function addMessage(source, message, category) {
  const type = classifyMessage(source);
  const el = document.createElement('div');
  el.className = `msg msg-${type}`;
  el.dataset.source = source;
  el.innerHTML = `<span class="speaker">[${source}]</span><span class="text">${escapeHtml(message)}</span>`;
  $messages.appendChild(el);

  // Trim old messages
  while ($messages.children.length > MAX_MESSAGES) {
    $messages.removeChild($messages.firstChild);
  }

  // Track last bot message for streaming
  if (type === 'bot') lastStreamMsg = el;

  scrollToBottom();
  resetScreensaverTimer();
}

function scrollToBottom() {
  $messages.scrollTop = $messages.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ── Status ─────────────────────────────────────────────────────────────────
function setStatus(status) {
  currentStatus = status;
  $statusText.textContent = status;

  // Remove old status classes, add new
  $status.className = `status-${status}`;
  $silenceFill.className = `fill-${status}`;
  $silenceFill.style.width = '0%';

  // Toolbar indicator
  $toolbarStatus.textContent = status === 'THINKING' ? '[PROCESSING]' : '[ACTIVE]';

  // Mic level animation
  if (status === 'LISTENING' || status === 'TALKING' || status === 'STANDBY') {
    micSilenceProgress = 0;
    startMicAnimation();
  } else {
    stopMicAnimation();
  }
}

// ── SocketIO events ────────────────────────────────────────────────────────

socket.on('kiosk_message', (data) => {
  addMessage(data.source, data.message, data.category);
});

socket.on('kiosk_stream', (data) => {
  // Update the last bot message in-place
  if (lastStreamMsg) {
    const textSpan = lastStreamMsg.querySelector('.text');
    if (textSpan) {
      textSpan.textContent = data.text;
      scrollToBottom();
    }
  }
  resetScreensaverTimer();
});

socket.on('kiosk_speaker_update', (data) => {
  // Find the message with old_speaker and matching text, update speaker
  const msgs = $messages.querySelectorAll('.msg');
  for (let i = msgs.length - 1; i >= 0; i--) {
    const msg = msgs[i];
    if (msg.dataset.source === data.old_speaker) {
      const textSpan = msg.querySelector('.text');
      if (textSpan && textSpan.textContent.includes(data.text.substring(0, 20))) {
        const speakerSpan = msg.querySelector('.speaker');
        speakerSpan.textContent = `[${data.new_speaker}]`;
        msg.dataset.source = data.new_speaker;
        break;
      }
    }
  }
});

socket.on('kiosk_status', (data) => {
  setStatus(data.status);
});

socket.on('kiosk_silence', (data) => {
  micSilenceProgress = data.progress;
  micSilenceMax = data.max || 1;

  if (data.max > 0 && data.progress > 0) {
    const pct = Math.min(100, (data.progress / data.max) * 100);
    $silenceFill.style.width = pct + '%';
  } else {
    $silenceFill.style.width = '0%';
  }

  // Ensure mic animation is running during listening
  if (currentStatus === 'LISTENING') startMicAnimation();
});

socket.on('kiosk_mic_level', (data) => {
  micLevel = data.level || 0;
});

socket.on('kiosk_think', () => {
  $toolbarStatus.textContent = '[PROCESSING]';
});

socket.on('kiosk_memory', () => {
  // Brief flash effect
  $toolbarStatus.textContent = '[MEMORY SAVE]';
  setTimeout(() => {
    $toolbarStatus.textContent = currentStatus === 'THINKING' ? '[PROCESSING]' : '[ACTIVE]';
  }, 1500);
});

socket.on('kiosk_wake', () => {
  deactivateScreensaver();
});

socket.on('kiosk_dnd', (data) => {
  if (data.enabled) {
    $dnd.classList.remove('hidden');
  } else {
    $dnd.classList.add('hidden');
  }
});

socket.on('kiosk_system', (data) => {
  if (data.battery) {
    const pct = data.battery.percentage;
    const chg = data.battery.charging;
    $battery.textContent = chg ? `${pct}%+` : `${pct}%`;
    $battery.className = pct > 50 ? '' : pct > 20 ? 'battery-mid' : 'battery-low';
  }
  if (data.cpu_temp !== undefined) {
    const temp = Math.round(data.cpu_temp);
    $cpuTemp.textContent = `${temp}C`;
    $cpuTemp.style.color = temp > 75 ? '#ff6400' : temp > 65 ? '#ffb400' : '';
  }
});

socket.on('kiosk_overlay', (data) => {
  $overlayImg.src = data.data;
  $overlayWrap.classList.remove('hidden');
  setTimeout(() => {
    $overlayWrap.classList.add('hidden');
  }, (data.duration || 8) * 1000);
});

// ── Power menu ─────────────────────────────────────────────────────────────
document.getElementById('btn-power').addEventListener('click', () => {
  $powerMenu.classList.toggle('hidden');
});
document.getElementById('btn-power-cancel').addEventListener('click', () => {
  $powerMenu.classList.add('hidden');
});
document.getElementById('btn-close').addEventListener('click', () => {
  fetch('/api/exit', { method: 'POST' }).catch(() => {});
  $powerMenu.classList.add('hidden');
});
document.getElementById('btn-shutdown').addEventListener('click', () => {
  fetch('/api/shutdown', { method: 'POST' }).catch(() => {});
  $powerMenu.classList.add('hidden');
});

// ── WiFi status ────────────────────────────────────────────────────────────
function updateWifi() {
  fetch('/api/wifi/status')
    .then(r => r.json())
    .then(d => {
      if (!d.connected) {
        $wifi.textContent = 'W:OFF';
        $wifi.className = 'wifi-off';
      } else {
        const signal = d.signal || 0;
        $wifi.textContent = `W:${signal}%`;
        $wifi.className = signal > 60 ? 'wifi-good' : signal > 30 ? 'wifi-fair' : 'wifi-poor';
      }
    })
    .catch(() => { $wifi.textContent = 'W:--'; $wifi.className = 'wifi-off'; });
}
updateWifi();
setInterval(updateWifi, 30000); // refresh every 30s

// ── Init ───────────────────────────────────────────────────────────────────
socket.on('connect', () => {
  console.log('Kiosk connected');
});
