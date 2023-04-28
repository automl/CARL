CARLify your Environment
========================

If you have an environment that you think would make an interesting addition
to CARL, feel free to add it. Here is a short guide on what to do:

1. Create a CARL wrapper class for your environment. It should be a child
class of CARLEnv and implement the setup and context changes of your environment.
   
2. Define the context. Provide a default context with all mutable features
as well as bounds for all features.
   
3. If additional requirements need to be installed, add them as install
options in setup.cfg. Make sure those changes are reflected in the ReadME
as well, especially if there are any extra steps (e.g. data downloads)
to the installation process.
   
4. Make a PR to the main branch of CARL. We'll review it and try to clarify
any open questions.
   
If you want to know if your environment would be a good fit or which context
features could be interesting to vary, don't hesitate to contact us!