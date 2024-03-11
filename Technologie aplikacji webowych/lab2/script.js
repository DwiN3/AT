// ZADANIE 1
console.log('\n-----------------Zadanie 1----------------');

const reverseStr = (str) => {
	var letters = str.split('');
	letters = letters.reverse();
	letters = letters.join('');
	return letters;
}

let words = "tsket ynocórwdo tsej oT"
console.log(words);
console.log(reverseStr(words));

// ZADANIE 2
console.log('\n-----------------Zadanie 2----------------');

const convertTemp = (temp, unitFrom, unitTo) => {
	if(unitFrom === unitTo){
		return temp;
	}
	if(unitFrom == "C"){
		temp = ((temp*9/5)+32);
	}
	else{
		temp = ((temp-32)/1.8);
	}
	return temp;
}

let cTemp = 10;
let fTemp = 50;
console.log("C: "+cTemp+"	F: "+ convertTemp(cTemp,"C","F"));
console.log("F: "+fTemp+"	C: "+ convertTemp(fTemp,"F","C"));
console.log("C: "+cTemp+"	C: "+ convertTemp(cTemp,"C","C"));

// ZADANIE 3
console.log('\n-----------------Zadanie 3----------------');

const sortTab = (numbers, type) => {
	if(type === "->"){
		return numbers = numbers.sort();
	}
	else{
		return numbers = numbers.sort().reverse();
	}
}

var numbers = [4,1,99,5,2];

console.log(sortTab(numbers, "->"));
console.log(sortTab(numbers, "<-"));