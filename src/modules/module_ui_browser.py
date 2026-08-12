"""
Module: Browser-based UI Manager

Lightweight UI manager that pushes all display state to a browser via
SocketIO instead of rendering with pygame.  A Chromium instance in kiosk
mode runs on the Pi's display, pointed at the local /kiosk page.

Implements the same public interface as UIManager / UIManagerLite so it
can be swapped in via config without changing any callers.
"""

import os
import subprocess
import threading
import time

from modules.module_messageQue import queue_message
from modules.module_config import load_config

CONFIG = load_config()


class UIManagerBrowser(threading.Thread):
    """UI manager that renders via a browser instead of pygame."""

    def __init__(self, shutdown_event, battery_module=None, cpu_temp_module=None, **kwargs):
        super().__init__(name="UIBrowserThread", daemon=True)
        self.shutdown_event = shutdown_event
        self.battery_module = battery_module
        self.cpu_temp_module = cpu_temp_module
        self.running = False
        self.paused = False
        self._chromium_proc = None
        self._socketio = None  # set lazily on first emit

    # -- SocketIO bridge --------------------------------------------------

    def _get_socketio(self):
        """Lazily fetch the SocketIO instance from module_chatui."""
        if self._socketio is None:
            try:
                from modules.module_chatui import socketio
                self._socketio = socketio
            except Exception:
                pass
        return self._socketio

    def _emit(self, event, data=None):
        """Emit a SocketIO event to the kiosk page."""
        sio = self._get_socketio()
        if sio:
            try:
                sio.emit(event, data or {})
            except Exception:
                pass

    # -- Lifecycle --------------------------------------------------------

    def run(self):
        """Main thread: launch Chromium kiosk and poll battery/cpu temp."""
        self.running = True
        self._launch_chromium()

        # Periodic status push (battery, cpu temp) every 5 seconds
        while self.running and not self.shutdown_event.is_set():
            if not self.paused:
                self._push_system_status()
            time.sleep(5)

    def stop(self):
        self.running = False
        self._kill_chromium()

    def _launch_chromium(self):
        """Launch Chromium in kiosk mode pointed at the local kiosk page."""
        port = CONFIG['ACCESS'].get('webui_port', 80)
        url = f"http://localhost:{port}/kiosk"

        # Wait for Flask to be ready
        for _ in range(30):
            try:
                import urllib.request
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(1)

        try:
            # Kill any existing Chromium instances
            subprocess.run(['pkill', '-f', 'chromium.*kiosk'], capture_output=True)
            time.sleep(0.5)

            env = os.environ.copy()
            env['DISPLAY'] = ':0'

            self._chromium_proc = subprocess.Popen(
                [
                    'chromium-browser',
                    '--kiosk',
                    '--noerrdialogs',
                    '--disable-infobars',
                    '--disable-session-crashed-bubble',
                    '--disable-translate',
                    '--no-first-run',
                    '--start-fullscreen',
                    '--autoplay-policy=no-user-gesture-required',
                    '--password-store=basic',
                    url,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            queue_message(f"LOAD: Browser kiosk launched at {url}")
        except FileNotFoundError:
            queue_message("WARNING: chromium-browser not found, trying chromium")
            try:
                self._chromium_proc = subprocess.Popen(
                    ['chromium', '--kiosk', '--noerrdialogs', '--disable-infobars',
                     '--no-first-run', '--start-fullscreen', url],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                queue_message(f"ERROR: Failed to launch browser: {e}")
        except Exception as e:
            queue_message(f"ERROR: Failed to launch browser: {e}")

    def _kill_chromium(self):
        if self._chromium_proc:
            try:
                self._chromium_proc.terminate()
                self._chromium_proc.wait(timeout=5)
            except Exception:
                try:
                    self._chromium_proc.kill()
                except Exception:
                    pass
            self._chromium_proc = None

    def _push_system_status(self):
        """Push battery and CPU temp to the kiosk page."""
        data = {}
        if self.battery_module and self.battery_module.sensor_initialized:
            try:
                status = self.battery_module.get_battery_status()
                data['battery'] = {
                    'percentage': status.get('normalized_percentage', 0),
                    'charging': status.get('is_charging', False),
                }
            except Exception:
                pass
        if self.cpu_temp_module:
            try:
                data['cpu_temp'] = self.cpu_temp_module.get_temperature()
            except Exception:
                pass
        if data:
            self._emit('kiosk_system', data)

    # -- Public interface (matches UIManager / UIManagerLite) --------------

    def update_data(self, source, message, category="INFO"):
        self._emit('kiosk_message', {
            'source': source,
            'message': message,
            'category': category,
        })

    def update_streaming_data(self, value):
        self._emit('kiosk_stream', {'text': value})

    def update_message_speaker(self, old_speaker, text, new_speaker):
        self._emit('kiosk_speaker_update', {
            'old_speaker': old_speaker,
            'text': text,
            'new_speaker': new_speaker,
        })

    def silence(self, progress=0):
        speechdelay = CONFIG['STT']['speechdelay']
        self._emit('kiosk_silence', {
            'progress': progress,
            'max': speechdelay,
        })

    def set_mic_level(self, level):
        self._emit('kiosk_mic_level', {'level': level})

    def set_tars_status(self, status):
        self._emit('kiosk_status', {'status': status})

    def think(self):
        self._emit('kiosk_think', {})

    def save_memory(self):
        self._emit('kiosk_memory', {})

    def deactivate_screensaver(self):
        self._emit('kiosk_wake', {})

    def set_dnd(self, enabled):
        self._emit('kiosk_dnd', {'enabled': enabled})

    def show_overlay_image(self, image_path, duration=8):
        # Convert to a URL the browser can fetch
        try:
            import base64
            with open(image_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(image_path)[1].lstrip('.')
            mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}.get(ext, 'image/png')
            self._emit('kiosk_overlay', {
                'data': f'data:{mime};base64,{b64}',
                'duration': duration,
            })
        except Exception:
            pass

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def exit_program(self):
        self.stop()
        os._exit(0)

    def initiate_shutdown(self):
        self.stop()
        self.shutdown_event.set()
        subprocess.run(['sudo', 'shutdown', 'now'])
