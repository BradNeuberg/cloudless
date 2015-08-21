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
    .attr({src: context.img_url})
    .load(function(){
      on_image_load(context);
    });
  $('.image-container').append(img);
}

function on_image_load(context){
  // Selection events will be relative to entire page; remove the influence of anything outside
  // our image container from our drawn bounding boxes.
  var image_offset = get_image_coords();

  var x_begin, y_begin, x_end, y_end;

  $('form').on('submit', on_submit);

  // Piggy-back on JQuery's selectable behavior to detect selection boxes.
  $('.image-container').selectable({
    distance: 5,
    filter: '#annotate-me',
    start: function(evt){
      // Get the mouse position when the user starts dragging.
      x_begin = evt.pageX - image_offset.left;
      y_begin = evt.pageY - image_offset.top;
    },

    stop: function(evt){
      // Get the mouse position when the user stops dragging.
      x_end = evt.pageX - image_offset.left;
      y_end = evt.pageY - image_offset.top;

      // If dragging mouse to the right, calculate the width & height.
      var drag_left = false;
      if ((x_end - x_begin) >= 1){
        width = x_end - x_begin;
        height = y_end - y_begin;
      }
      else {
        // We are dragging the mouse to the left.
        // TODO(neuberg): This doesn't seem to work consistently.
        width = x_begin - x_end;
        height = y_end - y_begin;
        drag_left = true;
      }

      // TODO(neuberg): Make sure we don't go beyond any of the edges of the image while drawing
      // the selection box.

      var box =
        $('<div></div>')
        .addClass('new-bbox')
        .css({
          left: x_begin,
          top: y_begin,
          width: width,
          height: height
        })
        .draggable({
          containment: '#annotate-me'
        })
        .resizable();

      // If the mouse was dragged left, offset our position.
      if (drag_left) {
        box.offset({
          left: x_end,
          top: y_begin
        });
      }

      $('.image-container').append(box);
    }
  });
}

function on_submit(evt){
  new_bboxes = [];
  $('.new-bbox')
    .each(function(idx, entry) {
      entry = $(entry);
      var pos = entry.position();
      new_bboxes.push([pos.left, pos.top, entry.width(), entry.height()]);
    });

  new_bboxes.forEach(function(box) {
    var input = $('<input type="hidden" name="new-bbox" />');
    input.val(box.join(','));
    $('form').append(input);
  });

  // TODO: Make sure there are either bounding boxes _or_ one of the checkboxes is selected.

  // TODO(neuberg): Remove once we have a real server.
  evt.preventDefault();
  print_debug_form();
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

/**
 * Calculates where our image is top, lef, width, and height in order to remove its effects from
 * the coordinates of newly drawn bounding boxes, as well as to prevent bounding boxes from being
 * dragged outside the image.
 */
function get_image_coords(){
  var img = $('#annotate-me');
  var offset = img.offset();

  // Coordinates can be floats on some platforms; we don't want floats to end up in our
  // drawn bounding boxes so normalize them now.
  var left = Math.round(offset.left);
  var top = Math.round(offset.top);

  return {
    left: left,
    top: top,
    width: img.width(),
    height: img.height()
  }
}

$(window).ready(init);
