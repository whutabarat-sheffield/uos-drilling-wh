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
2. Build the images
3. Run the images

#1
```
git clone https://github.com/whutabarat-sheffield/uos-drilling-wh.git .
```

#2
CPU version:
```
docker build -t listener.cpu -f Dockerfile.local.cpu .
docker run -t listener.cpu
```

GPU version:
```
docker build -t listener.gpu -f Dockerfile.local.gpu .
docker run -t listener.gpu
```

For testing in localhost:
```
docker build -t publisher -f Dockerfile.local.publisher .
docker run -t publisher
```

For testing with a Raspberry Pi (only on simple network):
```
docker build -t publisher -f Dockerfile.rpi.publisher .
docker run -t publisher
```