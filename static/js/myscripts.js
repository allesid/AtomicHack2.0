$(document).ready(function() {
    $(".owl-carousel").owlCarousel(
        $('.owl-carousel').owlCarousel({
            loop: true,
            margin: 10,
            nav: false,
            dots:false,
            responsive: {
                0: {
                    items: 2
                },
                600: {
                    items: 3
                },
                1000: {
                    items: 5
                }
            }
        })
    );
    let profilePic = document.getElementById("current-photo");

    inputFile.onchange = function (){
        profilePic.src = URL.createObjectURL(inputFile.files[0]);
    }
});
