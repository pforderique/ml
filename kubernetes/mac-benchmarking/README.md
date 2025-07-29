# Mac-benchmarking

Benchmark test using pytorch for Mac Mini M4.

## Prerequisites

1) have a mac mini M4, or just a mac with an M series chip
2) Install [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installing-from-release-binaries)
(Kubernetes in Docker) to run kubernetes locally

## Steps

1) Create a kubernetes cluster and make sure you are connected to it:
```bash
kind create cluster --name ml
kubectl config current-context
```

2) Build and push the image (locally)
```bash
docker build -t pforderique/pytorch-benchmark-cpu .
docker push pforderique/pytorch-benchmark-cpu:latest
```

3) Run the job and get output
```bash
kubectl apply -f job.yaml
kubectl get pods
kubectl logs pod/<pod_name>
```