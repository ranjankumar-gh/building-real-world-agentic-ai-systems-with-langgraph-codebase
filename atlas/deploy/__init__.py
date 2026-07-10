"""Chapter 22, "Deployment and Scaling" - the deploy-orchestration layer.

`rollout.py` holds `drain_replica`/`rolling_deploy`, the fleet-level
drain-migrate-deploy script built on top of Chapter 10's single-process
`RunControl.request_drain()`. `schedule.py` holds the Agent Server cron and
webhook wiring for Chapter 21's online quality monitor and the Chapter 17
research fan-out. Both are new in this chapter - Atlas had no deploy-time
orchestration code before Chapter 22.

`upgrade_check.py`, added in Chapter 26, "The Frontier and Future-Proofing",
holds `check_candidate_upgrade` - the upgrade-and-currency playbook that
points Chapter 21's unchanged regression suite at a candidate framework
version instead of an Atlas code change, wired to Chapter 22's stateless
cron (`schedule.py`) to run on a schedule rather than whenever someone
remembers."""
