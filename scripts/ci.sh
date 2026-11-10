#!/bin/sh
set -eu
gate_context=$(mktemp -d)
build_container=''
consumer_container=''
inspect_owned() {
  docker inspect -f '{"image":{{json .Image}},"user":{{json .Config.User}},"network":{{json .HostConfig.NetworkMode}},"binds":{{json .HostConfig.Binds}},"mounts":{{json .Mounts}},"cap_drop":{{json .HostConfig.CapDrop}},"security_opt":{{json .HostConfig.SecurityOpt}},"pids_limit":{{json .HostConfig.PidsLimit}},"memory":{{json .HostConfig.Memory}},"nano_cpus":{{json .HostConfig.NanoCpus}},"exit":{{json .State.ExitCode}},"oom_killed":{{json .State.OOMKilled}}}' "$1"
}
cleanup() {
  if [ -n "$build_container" ]; then docker rm -f "$build_container" >/dev/null 2>&1 || true; fi
  if [ -n "$consumer_container" ]; then docker rm -f "$consumer_container" >/dev/null 2>&1 || true; fi
  rm -rf -- "$gate_context"
}
trap cleanup EXIT HUP INT TERM
docker build -f Dockerfile.prepare -t calibrated-tabular-dependencies:local .
docker build --network none -f Dockerfile -t calibrated-tabular-test:local .
build_container=$(docker create --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 1g --cpus 2 calibrated-tabular-test:local)
docker start -a "$build_container"
test "$(docker inspect -f '{{.State.ExitCode}}' "$build_container")" = 0
mkdir -p "$gate_context/dist"
docker cp "$build_container":/work/distributions/. "$gate_context/dist/"
mkdir -p dist
docker cp "$build_container":/work/distributions/. dist/
inspect_owned "$build_container" > dist/build-container.json
docker cp "$build_container":/opt/wheelhouse "$gate_context/wheelhouse"
cp requirements-runtime.lock Dockerfile.consumer "$gate_context/"
cp scripts/consumer.py "$gate_context/consumer.py"
docker build --network none -f "$gate_context/Dockerfile.consumer" -t calibrated-tabular-consumer:local "$gate_context"
consumer_container=$(docker create --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 1g --cpus 2 calibrated-tabular-consumer:local)
docker start -a "$consumer_container"
test "$(docker inspect -f '{{.State.ExitCode}}' "$consumer_container")" = 0
mkdir -p dist/report-consumer
docker cp "$consumer_container":/work/report/. dist/report-consumer/
inspect_owned "$consumer_container" > dist/consumer-container.json
sha256sum dist/*.whl dist/*.tar.gz
printf '%s\n' 'OFFLINE_TESTS_AND_FRESH_INSTALLED_CONSUMER_COMPLETE'
