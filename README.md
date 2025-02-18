# uos-drilling
Latest version of depth estimation algorithm.
Please go into abyss for more details.

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
```
git clone https://github.com/whutabarat-sheffield/uos-drilling-wh.git .
```

Details on step #2 and #3
CPU version:
```
#  step #2 build the publisher and listener
docker build -t publisher -f Dockerfile.local.publisher .
docker build -t listener.cpu -f Dockerfile.local.cpu .
# step #3 run the publisher then listener
docker run -t publisher
docker run -t listener.cpu
```

GPU version:
```
# step #2 build the publisher and listener
docker build -t publisher -f Dockerfile.local.publisher .
docker build -t listener.gpu -f Dockerfile.local.gpu .
# step #3 run the publisher then listener
docker run -t publisher
docker run -t listener.gpu
```

For testing with a Raspberry Pi (still flaky and only on simple network):
```
docker build -t publisher -f Dockerfile.rpi.publisher .
docker run -t publisher
```



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
