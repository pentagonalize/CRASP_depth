#!/bin/bash

# Create directory for configs if it doesn't exist
mkdir -p configs

# Define the range of transformer layers to configure
min_transformer_layer=1
max_transformer_layer=10

# Define the range of languages (L) to include in train_test_sets
min_language=3
max_language=12

# Define common training and testing sets
train_set="100_to_150"
test_sets=("100_to_150" "150_to_200" "200_to_250" "250_to_300")
test_file_names=$(IFS=$',\n'; echo "${test_sets[*]}")

# Loop through the transformer layers
for transformer_layer in $(seq $min_transformer_layer $max_transformer_layer)
do
  # Start building the JSON content for this transformer layer
  json_content='{\n'
  json_content+='  "train_test_sets": {\n'

  # Loop through the languages and add to the train_test_sets
  for language_num in $(seq $min_language $max_language)
  do
    language_code="L${language_num}"
    json_content+="    \"${language_code}\": {\n"
    json_content+="      \"train_set\": \"${train_set}\",\n"
    json_content+="      \"test_sets\": [\""
    json_content+="${test_file_names//,/\",\"}"
    json_content+="\"]\n"
    json_content+="    }"
    if [[ "$language_num" -lt "$max_language" ]]; then
      json_content+=",\n"
    else
      json_content+='\n'
    fi
  done

  # Add the rest of the configuration for this transformer layer
  json_content+='  },\n'
  json_content+='  "heads": [\n    1\n  ],\n'
  json_content+='  "dims": [64,\n    256],\n'
  json_content+='  "lrs": [\n    0.00001,\n    0.000001\n  ],\n'
  json_content+="  \"epochs\": 10,\n"
  json_content+="  \"layers\": ${transformer_layer},\n"
  json_content+="  \"input_path\": \"data\",\n"
  json_content+="  \"output_path\": \"${transformer_layer}_layer/models\",\n"
  json_content+="  \"train_split\": 0.8,\n"
  json_content+="  \"bos_token\": 2\n"
  json_content+="}\n"

  # Write the JSON content to the config file for this transformer layer using printf -e
  printf -v json_escaped "%s" "$json_content"
  echo -e "$json_escaped" > configs/config_layer_${transformer_layer}.json

  # Create the output directory for this transformer layer
  mkdir -p "${transformer_layer}_layer/models"

  echo "Created configuration for transformer with ${transformer_layer} layer(s)"
done

echo "All configuration files generated in 'configs' directory"
