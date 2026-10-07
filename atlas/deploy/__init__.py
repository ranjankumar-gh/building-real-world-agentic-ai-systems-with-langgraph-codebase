"""Chapter 22, "Deployment and Scaling" - the deploy-orchestration layer.

`server.py` holds the graphs `langgraph.json` serves: the production
assembly (the mounted resolve agent and its gates, with the context built
from the authenticated identity) and the `monitor` graph, both compiled
without a checkpointer or store because the Agent Server supplies them.
`rollout.py` holds `drain_replica`/`rolling_deploy`, the fleet-level
drain-migrate-deploy script around the Agent Server's own SIGTERM drain
(Chapter 10's `RunControl.request_drain()` stays the mechanism for a run you
own in your own process). `schedule.py` holds the Agent Server cron and
webhook wiring for Chapter 21's online quality monitor and the Chapter 17
research fan-out. All three are new in this chapter - Atlas had no
deploy-time code before Chapter 22.

`upgrade_check.py`, added in Chapter 26, "The Frontier and Future-Proofing",
holds `check_candidate_upgrade` - the upgrade-and-currency playbook that
points Chapter 21's unchanged regression suite at a candidate framework
version instead of an Atlas code change, wired to Chapter 22's stateless
cron (`schedule.py`) to run on a schedule rather than whenever someone
remembers."""
