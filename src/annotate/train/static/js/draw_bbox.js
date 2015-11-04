var new_bboxes = [];

function init(){
  var context = window.annotation_context;

  $('input[name="image_id"]').val(context.image_id);

  // Add the image we are working with to the page.
  var img =
    $('<img id="annotate-me" />')
    .on('dragstart', function(evt){
      // Turn off the browser's native image dragging behavior, as it will get in the way
      // of drawing our bounding boxes.
      evt.preventDefault();
    })
    .attr({src: context.image_url})
    .load(function(){
      on_image_load(context);
    });
  $('.image-container').append(img);
}

function on_image_load(context){
  $('form').on('submit', on_submit);

  $('#annotate-me').selectAreas({
    overlayOpacity: 0.0
  });
}

function on_submit(evt){
  var areas = $('#annotate-me').selectAreas('relativeAreas');

  areas.forEach(function(box) {
    var input = $('<input type="hidden" name="new-bbox" />');
    input.val([box.x, box.y, box.width, box.height].join(','));
    $('form').append(input);
  });

  // TODO: Make sure there are either bounding boxes _or_ one of the checkboxes is selected.

  // Uncomment for debugging:
  //evt.preventDefault();
  //print_debug_form();
}

/**
 * Utility function to print out our HTML input values after a user hits submit.
 */
function print_debug_form(){
  console.log('Submitted form results:');
  $('input').each(function(idx, elem) {
    elem = $(elem);

    if (elem.attr('type') == 'checkbox' && elem[0].checked) {
      console.log(elem.attr('name') + '=' + elem.val());
    } if (elem.attr('type') !== 'checkbox') {
      console.log(elem.attr('name') + '=' + elem.val());
    }
  });
}

$(window).ready(init);
