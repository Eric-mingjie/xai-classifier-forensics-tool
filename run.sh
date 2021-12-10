#!/bin/bash
#SBATCH --job-name=flask
#SBATCH --output=flask.out
#SBATCH --gres=gpu:3
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --exclude=locus-0-16
export FLASK_APP=main.py  
export FLASK_ENV=development
export PORT=8080
/usr/bin/ssh -N -f -R $PORT:localhost:$PORT locus.eth
flask run --port $PORT
