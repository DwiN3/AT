function resetErrors() {
    $("label").removeClass("error");
}

function showErrors(errorFields) {
    errorFields.forEach(function(item) {
        $('#'+item + "-label").addClass("error");
    });
}

function registerUser() {
    var user = {
        login: $("#login").val(),
        firstName: $("#firstName").val(),
        lastName: $("#lastName").val(),
        password: $("#password").val(),
        pesel: $("#pesel").val()
    };

    resetErrors();

    $.ajax({
        url: '/api/users/register',
        type: 'post',
        contentType: 'application/json',
        success: function (data) {
            console.log('data: ', data);
            $('#target').html("<p><b class='ok'>Dane zostały zapisane na serwerze :-)</b></p>");
        },
        error: function(jqXhr, textStatus, errorThrown) {
            console.log('error: ', jqXhr.responseText.split(","));

            $('#target').html("<p><b class='error'>Formularz zawiera błędy!</b></p>");
            var errorFields = jqXhr.responseText.split(",");
            showErrors(errorFields)
        },
        data: JSON.stringify(user)
    });
}

$(document).ready(function() {

    $("#register-user").submit(function(e) {
        e.preventDefault();
        registerUser();
    });
});
