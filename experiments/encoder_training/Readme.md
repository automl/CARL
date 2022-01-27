# Training a Context Encoder

We provide pretrained encoders in 'database', but you can also train your own using this code. To do so, run:
```bash
python train_encoder.py '+environment=pendulum' '+encoder.context_dataset=./database/Pendulum/new_60k.npy'
```

If you want to use an alternative context base, you can sample one like this:
```bash
python create_context_database.py
```
