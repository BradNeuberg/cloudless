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
  $('.delete').on('click', on_delete);
  $('form').on('submit', on_submit);

  $('#annotate-me').selectAreas({
    overlayOpacity: 0.0
  });
}

function on_delete(evt){
  var delete_me = $('<input type="hidden" name="delete" value="true" />');
  $('form').append(delete_me);
}

function on_submit(evt){
  var areas = $('#annotate-me').selectAreas('relativeAreas');

  if ($('input.all-cloud').prop('checked')) {
    var input = $('<input type="hidden" name="new-bbox" />');
    input.val([0, 0, 512, 512].join(','));
    $('form').append(input);
  } else {
    areas.forEach(function(box) {
      var input = $('<input type="hidden" name="new-bbox" />');
      input.val([box.x, box.y, box.width, box.height].join(','));
      $('form').append(input);
    });
  }

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
