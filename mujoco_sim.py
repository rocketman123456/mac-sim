import mujoco
import mujoco.viewer


# Load the model
# spec = mujoco.MjSpec()
model = mujoco.MjModel.from_xml_path("unitree_go1/scene.xml")
data = mujoco.MjData(model)

model.opt.timestep = 0.002
key_id = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, key_id)
pause_flag = True


def key_callback(keycode):
    viewer.lock()
    if chr(keycode) == " ":
        global pause_flag
        pause_flag = not pause_flag


viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

# Run a viewer to interact with the simulation
while viewer.is_running():
    if pause_flag:
        mujoco.mj_forward(model, data)
    else:
        mujoco.mj_step(model, data)
    viewer.sync()
