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

const $micCanvas  = document.getElementById('mic-level');
const micCtx      = $micCanvas.getContext('2d');

let currentStatus = 'BOOTING';
let screensaverActive = false;
let screensaverTimer = null;
let lastStreamMsg = null;   // reference to the last bot message element (for streaming)
const SCREENSAVER_TIMEOUT = 300000; // 5 minutes
const MAX_MESSAGES = 50;

// Mic level state
let micActive = false;
let micSilenceProgress = 0;
let micSilenceMax = 1;
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

  if (currentStatus !== 'LISTENING' && currentStatus !== 'TALKING') {
    micAnimFrame = null;
    return;
  }

  const barCount = 40;
  const barWidth = w / barCount;
  const isListening = currentStatus === 'LISTENING';
  const isTalking = currentStatus === 'TALKING';

  for (let i = 0; i < barCount; i++) {
    let level;
    if (isTalking) {
      // Smooth wave animation when TARS is talking
      level = 0.3 + 0.7 * Math.abs(Math.sin((Date.now() / 200) + i * 0.3));
    } else if (micSilenceProgress === 0) {
      // Speech detected — active random bars
      level = 0.3 + Math.random() * 0.7;
    } else {
      // Silence — bars decay based on progress
      const decay = 1 - (micSilenceProgress / Math.max(1, micSilenceMax));
      level = Math.max(0.05, decay * (0.2 + Math.random() * 0.3));
    }

    const barH = level * h;
    const y = (h - barH) / 2;

    // Color: cyan when active, dims toward dark as silence grows
    const alpha = isListening ? (micSilenceProgress === 0 ? 0.9 : 0.3 + 0.6 * (1 - micSilenceProgress / Math.max(1, micSilenceMax))) : 0.7;
    micCtx.fillStyle = isTalking
      ? `rgba(0, 255, 100, ${alpha})`
      : `rgba(0, 255, 255, ${alpha})`;
    micCtx.fillRect(i * barWidth + 1, y, barWidth - 2, barH);
  }

  micAnimFrame = requestAnimationFrame(drawMicLevel);
}

function startMicAnimation() {
  if (!micAnimFrame) {
    micAnimFrame = requestAnimationFrame(drawMicLevel);
  }
}

function stopMicAnimation() {
  if (micAnimFrame) {
    cancelAnimationFrame(micAnimFrame);
    micAnimFrame = null;
  }
  micCtx.clearRect(0, 0, $micCanvas.width, $micCanvas.height);
}

// ── Clock ──────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const h = now.getHours();
  const m = String(now.getMinutes()).padStart(2, '0');
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h % 12 || 12;
  $toolbarTime.textContent = `${h12}:${m} ${ampm}`;

  if (screensaverActive) {
    $screensaverClock.textContent = `${h12}:${m} ${ampm}`;
  }
}
setInterval(updateClock, 10000);
updateClock();

// ── Screensaver ────────────────────────────────────────────────────────────
let starCtx, stars = [];

function initStars() {
  const canvas = $screensaverCanvas;
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  starCtx = canvas.getContext('2d');
  stars = [];
  for (let i = 0; i < 150; i++) {
    stars.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      size: Math.random() * 2 + 0.5,
      speed: Math.random() * 0.3 + 0.1,
      alpha: Math.random(),
    });
  }
}

function drawStars() {
  if (!screensaverActive) return;
  const c = $screensaverCanvas;
  starCtx.fillStyle = '#000';
  starCtx.fillRect(0, 0, c.width, c.height);
  for (const s of stars) {
    s.alpha += (Math.random() - 0.5) * 0.05;
    s.alpha = Math.max(0.2, Math.min(1, s.alpha));
    s.y += s.speed;
    if (s.y > c.height) { s.y = 0; s.x = Math.random() * c.width; }
    starCtx.fillStyle = `rgba(0,255,255,${s.alpha})`;
    starCtx.beginPath();
    starCtx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
    starCtx.fill();
  }
  requestAnimationFrame(drawStars);
}

function activateScreensaver() {
  if (screensaverActive) return;
  screensaverActive = true;
  $screensaver.classList.remove('hidden');
  initStars();
  drawStars();
  updateClock();
}

function deactivateScreensaver() {
  screensaverActive = false;
  $screensaver.classList.add('hidden');
  resetScreensaverTimer();
}

function resetScreensaverTimer() {
  clearTimeout(screensaverTimer);
  screensaverTimer = setTimeout(activateScreensaver, SCREENSAVER_TIMEOUT);
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
  if (status === 'LISTENING' || status === 'TALKING') {
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

// ── Init ───────────────────────────────────────────────────────────────────
socket.on('connect', () => {
  console.log('Kiosk connected');
});
