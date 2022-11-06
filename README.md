A very early WIP of an embeddings editor for AUTOMATIC1111's webui

Should be placed in a sub-directory in extensions. e.g. \stable-diffusion-webui-master\extensions\embedding-editor\

It will likely add a small amount of startup time due to fetching all the original embeddings to calculate the ranges for the weight sliders, so it's probably better to only enable when you're using it for now

It currently doesn't check for invalid inputs and will likely error easily (e.g. you can enter invalid vector numbers, or click save embedding while no embedding is selected)

'Refresh Embeddings' will repopulate the embeddings list, though won't reset the weight sliders on the currently selected embedding. Select a different embedding or change the vector number to refresh the weight sliders

Changing the vector number will lose any edited weights for the current vector. Saving the embedding first is required if you want to keep them

It's not easy to get any useful results with this yet, though some features are planned which might help a lot
