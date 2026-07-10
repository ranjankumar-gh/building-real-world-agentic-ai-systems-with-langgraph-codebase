"""Chapter 22, "Deployment and Scaling" - the deploy-orchestration layer.

`rollout.py` holds `drain_replica`/`rolling_deploy`, the fleet-level
drain-migrate-deploy script built on top of Chapter 10's single-process
`RunControl.request_drain()`. `schedule.py` holds the Agent Server cron and
webhook wiring for Chapter 21's online quality monitor and the Chapter 17
research fan-out. Both are new in this chapter - Atlas had no deploy-time
orchestration code before Chapter 22."""
