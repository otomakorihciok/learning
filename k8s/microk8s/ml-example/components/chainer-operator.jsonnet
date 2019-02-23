local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["chainer-operator"];

local k = import "k.libsonnet";
local operator = import "kubeflow/chainer-job/chainer-operator.libsonnet";

local namespace = env.namespace;  // namespace is inherited from the environment
local name = params.name;
local image = params.image;
local serviceAccountName = if params.createRbac != "true" then params.serviceAccountName else name;
local v = params.v;
local stderrthreshold = params.stderrthreshold;

if params.createRbac == "true" then
  std.prune(k.core.v1.list.new([
    operator.parts.crd,
    operator.parts.clusterRole(name),
    operator.parts.serviceAccount(namespace, name),
    operator.parts.clusterRoleBinding(namespace, name),
    operator.parts.deploy(namespace, name, image, serviceAccountName, v, stderrthreshold),
  ]))
else
  std.prune(k.core.v1.list.new([
    operator.parts.crd,
    operator.parts.deploy(namespace, name, image, serviceAccountName, v, stderrthreshold),
  ]))
