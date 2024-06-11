function isCorrectMove(source, destination, type) {
    const figure = {
        source: source,
        destination: destination,
        figureType: type
    };

    return $.ajax({
        url: '/api/chess/is-correct-move',
        type: 'post',
        contentType: 'application/json',
        success: function (data) {
            console.log('success...');
            return data;
        },
        error: function() {
            console.log("error...");
            alert("Wystąpił nieoczekiwany problem z usługą!");
            return false;
        },
        data: JSON.stringify(figure)
    });
}

$(document).ready(function() {
    var chooseFigure = 'BISHOP';

    var figures = new Map();
    figures.set('KING', '<span id="figure">&#9812;</span>');
    figures.set('QUEEN', '<span id="figure">&#9813;</span>');
    figures.set('ROOK', '<span id="figure">&#9814;</span>');
    figures.set('BISHOP', '<span id="figure">&#9815;</span>');
    figures.set('KNIGHT', '<span id="figure">&#9816;</span>');
    figures.set('PAWN', '<span id="figure">&#9817;</span>');

    var startPosition = null;
    var destinationPosition = null;

    $("#chess").change(function() {
        console.log('value = ', $(this).val());
        chooseFigure = $(this).val();

        $('#figure').remove();
        switch(chooseFigure) {
            case 'KING':  $('#e_1').html(figures.get(chooseFigure)); break;
            case 'QUEEN': $('#d_1').html(figures.get(chooseFigure)); break;
            case 'ROOK':  $('#a_1').html(figures.get(chooseFigure)); break;
            case 'BISHOP':  $('#c_1').html(figures.get(chooseFigure)); break;
            case 'KNIGHT':  $('#b_1').html(figures.get(chooseFigure)); break;
            case 'PAWN':  $('#d_2').html(figures.get(chooseFigure)); break;
        }
    });

    $(".field").mouseup(function(){
        console.log('mousedown = ' + $(this).attr('id'));
        console.log('mousedown = ' + $(this).find('#figure').length);

        if($(this).find('#figure').length === 1 && startPosition == null) {
            startPosition = $(this).attr('id');
            $('#figure').css('color', '#267340');

        } else if(startPosition != null) {
            destinationPosition = $(this).attr('id');

            var resultMove = isCorrectMove(startPosition, destinationPosition, chooseFigure);

            resultMove.then(function(response) {
                if(response) {
                    $('#figure').css('color', '#000000');

                    $('#figure').remove();
                    $('#'+destinationPosition).html(figures.get(chooseFigure));

                } else {
                    $('#figure').css('color', '#000000');

                    $('#figure').remove();
                    $('#'+startPosition).html(figures.get(chooseFigure));
                }

                startPosition = null;
                destinationPosition = null;
            });
        }
    });
})