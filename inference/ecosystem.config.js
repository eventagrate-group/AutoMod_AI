module.exports = {
    apps: [{
      name: 'toxictext-scanner',
      script: './venv/bin/gunicorn',
      args: '--workers=4 --bind=0.0.0.0:5001 app:app',
      interpreter: 'python3',
      env: {
        NLTK_DATA: './nltk_data'
      },
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      max_memory_restart: '1G'
    }]
  }