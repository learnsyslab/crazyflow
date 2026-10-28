# Contact examples

Querying contacts between drones and the scene, and choosing the collision geometry of the drone. See [MuJoCo Integration](../user-guide/mujoco.md) for how the scene is built.

---

## Contact queries

--8<-- example "examples/contacts/contacts.py" media="video" cam_config="distance=2.2 azimuth=110 elevation=-25 lookat=-0.1,0,0.35" geomgroup="0,1,2,3"

---

## Crash detection

--8<-- example "examples/contacts/crash.py" media="video" camera="track_cam:0"

---

## Collision pairs

--8<-- example "examples/contacts/collision_pairs.py" media="video,output" camera="track_cam:0"
