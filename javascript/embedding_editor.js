function embedding_editor_update_guidance(col_weights) {
	let weightsColsVals = {};
	for (let i=0; i<768; i++)
		weightsColsVals[i] = [];
	
	for (let color in col_weights) {
		let weights = Object.values(col_weights[color]);
		
		for (let i=0; i<768; i++)
			embedding_editor_sorted_insert(weightsColsVals[i], color, weights[i]);
	}
	
	for (let i=0; i<768; i++) {
		let guidanceElem = embedding_editor_get_guidance_bar(i),
			variants = weightsColsVals[i];
		
		// 	e.g. linear-gradient(to right, transparent 0%, transparent 31%, red 31%, red 34%, transparent 34%, transparent 100%)
		let transparent = '#00000000',
			bg = 'linear-gradient(to right',
			previous = 0;
		
		for (let j=0, jLen=variants.length; j<jLen; j++) {
			let variant = variants[j],
				color = variant.color,
				weight = variant.weight * 100, // currently in decimal format
				start = Math.max(previous, weight-2),
				end = Math.max(previous, weight+2);
			
			if (previous < start)
				bg += ', ' + transparent + ' ' + previous + '%';
				bg += ', ' + transparent + ' ' + start + '%';
			
			bg += ', ' + color + ' ' + Math.max(previous, weight-1) + '%';
			bg += ', ' + color + ' ' + Math.max(previous, weight+1) + '%';
			
			previous = end;
		}
		
		bg += ', ' + transparent + ' ' + previous + '%';
		bg += ', ' + transparent + ' 100%)';
		
		guidanceElem.style.background = bg;
	}
}

function embedding_editor_sorted_insert(set, color, weight) {
	let insertAt = 0;
	for (let len=set.length; insertAt<len; insertAt++)
		if (set[insertAt].weight > weight)
			break;
	
	set.splice(insertAt, 0, { color : color, weight: weight });
}

function embedding_editor_get_guidance_bar(weightNum) {
	let guidanceClassType = "embedding_editor_guidance_bar",	
		gradioSlider = document.querySelector("gradio-app").shadowRoot.getElementById("embedding_editor_weight_slider_" + weightNum),
		childElems = Array.from(gradioSlider.childNodes),
		lastChild = childElems[childElems.length-1];
	
	if (!lastChild.classList.contains(guidanceClassType)) {
		// currently pointing to the range input. Move it slightly lower to line up with the new div, then insert the new div
		lastChild.style.verticalAlign = 'text-bottom'; // could do this with CSS, have the selector for it, and then won't change on first time pressing button
		
		let newElem = document.createElement("div");
		newElem.style.height='6px';
		newElem.classList.add(guidanceClassType);
		gradioSlider.appendChild(newElem);
		lastChild = newElem;
	}
	
	return lastChild; // could just cache these
}

onUiUpdate(function(){
    
})
