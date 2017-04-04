---
layout: post
title:  "How to use Docker"
date:   2017-04-04 14:16:00 +0200
categories: utilities
---

Images vs Containers
====================

According to [this post][docker-image-vs-container] 
> An image is an inert, immutable, file that's essentially a snapshot of a container. 
> Images are created with the build command, and they'll produce a container when started with run. 
> Images are stored in a Docker registry such as registry.hub.docker.com. 
> Because they can become quite large, images are designed to be composed of layers of other images, allowing a miminal amount of data to be sent when transferring images over the network.

{% highlight bash %}
# List local images
docker images

# List running containers
docker ps

# List all containers
docker ps -a

# Delete all containers
docker rm $(docker ps -a -q)

# Delete all images
docker rmi $(docker images -q)
{% endhighlight %}

Start a new container
=====================

{% highlight bash %}
# Pull the image called 'ocaml/opam:alpine'
docker pull ocaml/opam:alpine

# Run new instance of the image just downloaded
docker run -i -t ocaml/opam:alpine /bin/bash
# -a=[]           : Attach to `STDIN`, `STDOUT` and/or `STDERR`
# -t              : Allocate a pseudo-tty
# --sig-proxy=true: Proxy all received signals to the process (non-TTY mode only)
# -i              : Keep STDIN open even if not attached

# Run new instance with volume
docker run -v <host dir>:<guest dir> <image name> <command>
docker run -v C:/mydirec:/guestdirec alpine       ls /data

# Run new instance (full command)
docker run -i -t -v c:/DockerDataIncud:/data ocaml/opam:alpine /bin/bash
{% endhighlight %}

[docker-image-vs-container]: http://paislee.io/how-to-automate-docker-deployments/