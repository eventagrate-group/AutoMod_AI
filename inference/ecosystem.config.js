module.exports = {
    apps: [{
      name: 'toxictext-scanner',
      script: 'venv/bin/gunicorn',
      args: '--workers=4 --bind=0.0.0.0:5001 app:app',
      cwd: '/home/branch/projects/toxictext-scanner/inference',
      interpreter: 'python3',
      env: {
        NLTK_DATA: '/home/branch/projects/toxictext-scanner/inference/nltk_data'
      },
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      max_memory_restart: '1G'
    }]
  }