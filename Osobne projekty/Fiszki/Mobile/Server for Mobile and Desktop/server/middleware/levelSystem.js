module.exports = (result, usersPoints, requiredPoints, level) => {

    let parsedRequiredPoints = parseInt(requiredPoints);
    let parsedLevel = parseInt(level);
    let points = parseInt(result) + parseInt(usersPoints);

    if(points >= requiredPoints){
        while(points >= parsedRequiredPoints){
            parsedLevel++;
            points -= parsedRequiredPoints;
            parsedRequiredPoints += 200; 
        }
    }
    const summary = {
        points : points,
        requiredPoints : parsedRequiredPoints,
        level : parsedLevel
    }     
    return summary;    
}