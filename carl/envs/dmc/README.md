# Headless Rendering
If you have problems with OpenGL, this helped:
Set this in your script
```python
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'                                                                                                          
os.environ['MUJOCO_GL'] = 'osmesa'                                                                                                                    
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'            
```

And set ErrorChecker to None in `OpenGL/raw/GL/_errors.py`.