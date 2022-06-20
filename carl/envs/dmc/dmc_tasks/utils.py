from lxml import etree


def adapt_context(xml_string, context):
  """Adapts and returns the xml_string of the model with the given context."""
  print(context)
  mjcf = etree.fromstring(xml_string)
  damping = mjcf.find("./default/joint")
  damping.set("damping", str(context["joint_damping"]))
  geom = mjcf.find("./default/geom")
  geom.set("friction", " ".join([
    str(context["friction_tangential"]), 
    str(context["friction_torsional"]), 
    str(context["friction_rolling"])])
  )
  geom_density = geom.get("density")
  if not geom_density:
    geom_density = 1000
  geom.set("density", str(geom_density * context["geom_density"]))
  actuators = mjcf.findall("./actuator/motor")
  for actuator in actuators:
    gear = actuator.get("gear")
    actuator.set("gear", str(int(float(gear) * context["actuator_strength"])))
  keys = []
  options = mjcf.findall("./option")
  gravity = " ".join(["0", "0", str(context["gravity"])])
  wind = " ".join([str(context["wind_x"]), str(context["wind_y"]), str(context["wind_z"])])
  for option in options:
    for k, _ in option.items():
      keys.append(k)
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
      
  if "gravity" not in keys:
    mjcf.append(etree.Element("option", gravity=gravity))
  if "timestep" not in keys:
    mjcf.append(etree.Element("option", timestep=str(context["timestep"])))
  if "wind" not in keys:
    mjcf.append(etree.Element("option", wind=wind))
  if "density" not in keys:
    mjcf.append(etree.Element("option", density=str(context["density"])))
  if "viscosity" not in keys:
    mjcf.append(etree.Element("option", viscosity=str(context["viscosity"])))
  xml_string = etree.tostring(mjcf, pretty_print=True)
  print(xml_string.decode("utf-8"))
  return xml_string