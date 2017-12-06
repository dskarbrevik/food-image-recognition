import coremltools

coreml_model = coremltools.converters.keras.convert('./model4b.10-0.68.hdf5', input_names=['image'], output_names=['foodConfidence'], class_labels='labels.txt', image_input_names=['image'], image_scale=2/255.0, red_bias=-1, green_bias=-1, blue_bias=-1)
coreml_model.license = 'Apache 2.0'
coreml_model.short_description = 'This is a coreml model from the food101 model'
coreml_model.save('converted.mlmodel')


