"""Double-click launcher: isolated setup, health check, then browser."""
from pathlib import Path
import hashlib, os, socket, subprocess, sys, time, urllib.request, venv, webbrowser
ROOT=Path(__file__).resolve().parent

def main():
    os.chdir(ROOT)
    env=ROOT/'.venv'
    python=env/'Scripts'/'python.exe'
    if not python.exists():
        print('Creating the local Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    requirements=ROOT/'requirements.txt'
    fingerprint=hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker=env/'requirements.sha256'
    if not marker.exists() or marker.read_text()!=fingerprint:
        print('Installing dependencies. First setup can take several minutes and needs internet.',flush=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(requirements)],check=True)
        subprocess.run([str(python),'-c','import streamlit, ultralytics, cv2, lap, imageio_ffmpeg'],check=True)
        marker.write_text(fingerprint)
    # Pick an unused local port; do not expose the app to the network.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    print('Starting app:',url,flush=True)
    print('Keep this window open. Press Ctrl+C here to stop the app.',flush=True)
    log=ROOT/'startup.log'
    with log.open('w',encoding='utf-8') as output:
        proc=subprocess.Popen([str(python),'-m','streamlit','run',str(ROOT/'app.py'),
            '--server.address=127.0.0.1',f'--server.port={port}',
            '--server.headless=true','--browser.gatherUsageStats=false'],stdout=output,stderr=subprocess.STDOUT)
        try:
            ready=False
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(120):
                if proc.poll() is not None: raise RuntimeError('App exited during startup; see startup.log.')
                try:
                    with opener.open(url+'/_stcore/health',timeout=1) as response:
                        ready=response.status==200
                except OSError: pass
                if ready: break
                time.sleep(0.5)
            if not ready: raise RuntimeError('App startup timed out; see startup.log.')
            webbrowser.open(url)
            print('App ready. If the browser did not open, paste this address:',url,flush=True)
            return proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()

if __name__=='__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: print('App stopped.'); sys.exit(0)
    except Exception as exc:
        print('Startup failed:',exc)
        print('Check your internet connection and the messages above. App logs are in startup.log.')
        sys.exit(1)
