from mcunet.model_zoo import build_model
model, image_size, description = build_model(net_id="mcunet-in2", pretrained=True)  # you can replace net_id with any other option from net_id_list
print(image_size)