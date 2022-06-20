from wsgiref.simple_server import demo_app
from lxml import etree
from torch import ge


def adapt_context(xml_string, context):
  """Adapts and returns the xml_string of the model with the given context."""
  mjcf = etree.fromstring(xml_string)
  default = mjcf.find("./default/")
  if default is None:
    default = etree.Element("default")
    mjcf.addnext(default)

  # adjust damping for all joints if damping is already an attribute
  for joint_find in mjcf.findall(".//joint[@damping]"):
    joint_damping = joint_find.get("damping")
    joint_find.set("damping", str(float(joint_damping) * context["joint_damping"]))

  # adjust stiffness for all joints if stiffness is already an attribute
  for joint_find in mjcf.findall(".//joint[@stiffness]"):
    joint_stiffness = joint_find.get("stiffness")
    joint_find.set("stiffness", str(float(joint_stiffness) * context["joint_stiffness"]))

  # set default joint damping if default/joint is not present
  joint = mjcf.find("./default/joint")
  if joint is None:
    joint = etree.Element("joint")
    default.addnext(joint)
    def_joint_damping = 0.1
    default_joint_damping = str(float(def_joint_damping) * context["joint_damping"])
    joint.set("damping", default_joint_damping)
    default_joint_stiffness = str(context["joint_stiffness"])
    joint.set("stiffness", default_joint_stiffness)

  # adjust friction for all geom elements with friction attribute
  for geom_find in mjcf.findall(".//geom[@friction]"):
    friction = geom_find.get("friction").split(" ")
    frict_str = ""
    for f, d in zip(friction, [context["friction_tangential"]*2, context["friction_torsional"], context["friction_rolling"]]):
      frict_str += str(float(f) * d) + " "
    geom_find.set("friction", frict_str[:-1])

  # adjust density for all geom elements with density attribute
  for geom_find in mjcf.findall(".//geom[@density]"):
    geom_find.set("density", str(float(geom_find.get("density")) * context["geom_density"]))

  # create default geom if it does not exist
  geom = mjcf.find("./default/geom")
  if geom is None:
    geom = etree.Element("geom")
    default.addnext(geom)

  # set default friction
  if geom.get("friction") is None:
    default_friction_tangential = 1.
    default_friction_torsional = 0.005
    default_friction_rolling = 0.0001
    geom.set("friction", " ".join([
      str(default_friction_tangential * context["friction_tangential"]), 
      str(default_friction_torsional * context["friction_torsional"]), 
      str(default_friction_rolling * context["friction_rolling"])])
    )

  # set default density
  geom_density = geom.get("density")
  if geom_density is None:
    geom_density = 1000
    geom.set("density", str(float(geom_density) * context["geom_density"]))

  actuators = mjcf.findall("./actuator/")
  for actuator in actuators:
    gear = actuator.get("gear")
    if gear is None:
      gear = 1
    actuator.set("gear", str(float(gear) * context["actuator_strength"]))


  # find option settings and override them if they exist, otherwise create new option
  option_keys = []
  options = mjcf.findall(".//option")
  gravity = " ".join(["0", "0", str(context["gravity"])])
  wind = " ".join([str(context["wind_x"]), str(context["wind_y"]), str(context["wind_z"])])
  for option in options:
    for k, _ in option.items():
      option_keys.append(k)
      if k == "gravity":
        option.set("gravity", gravity)
      elif k == "timestep":
        option.set("timestep", str(context["timestep"]))
      elif k == "density":
        option.set("density", str(context["density"]))
      elif k == "viscosity":
        option.set("viscosity", str(context["viscosity"]))
      elif k == "wind":
        option.set("wind", wind)
  if "gravity" not in option_keys:
    mjcf.append(etree.Element("option", gravity=gravity))
  if "timestep" not in option_keys:
    mjcf.append(etree.Element("option", timestep=str(context["timestep"])))
  if "wind" not in option_keys:
    mjcf.append(etree.Element("option", wind=wind))
  if "density" not in option_keys:
    mjcf.append(etree.Element("option", density=str(context["density"])))
  if "viscosity" not in option_keys:
    mjcf.append(etree.Element("option", viscosity=str(context["viscosity"])))

  xml_string = etree.tostring(mjcf, pretty_print=True)
  return xml_string
