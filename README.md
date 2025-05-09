# uos-drilling
Latest version of depth estimation algorithm.
Please go into the release page for more up-to-date details.

## Quick installation
Clone the repository

Change into /abyss

Type 'pip install .'

Go into test and run it

# Docker installation

1. Clone the entire repository
2. Build the Docker images for publisher and listener
3. Run the Docker images

Details on step #1
```bash
git clone https://github.com/whutabarat-sheffield/uos-drilling-wh.git .
```

Details on step #2 and #3
```bash
#  step #2 build the listener
docker build -t listener .
```

```
# step #3 run the listener
docker run -t listener
```

The listener is callable from the command line as 
```bash 
uos_depthest_listener --config (path to the configuration file)
```

The docker build will already setup a workspace under `/app` directory, which contains the configuration file `mqtt_conf.yaml`. 


Below are Windo's notes on how to manage Docker -- please ignore:
```
# Building
docker build -t uos-listener-test001 .
# Debugging
docker run --rm -it --entrypoint /bin/bash uos-listener-test001
# Checking site-packages
cd /usr/local/lib/python3.10/site-packages/abyss
# Running the publisher
docker build -f Dockerfile.publisher -t uos-publisher-test001 .
# Cleaning unused containers
docker system prune -a 
```
