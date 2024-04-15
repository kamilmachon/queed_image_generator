# Queed NFT generator

install python3.10

install venv: 'python -m pip install --user virtualenv'

create virtual envirement: 'python3 -m venv .venv'

source venv: 'source .venv/bin/activate'

install torch: 'pip install torch --index-url https://download.pytorch.org/whl/cu117'

install requirements: 'pip install -r requirements.txt'

fill config file with fetch ai and S3 configuration

launch agent 'python image_generating_agent.py --model cpu' or 'python image_generating_agent.py --model gpu'
