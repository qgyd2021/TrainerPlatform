

var get_available_models = function(callback){
  var url = "basic_intent/get_available_models";

  $.ajax({
    type: "POST",
    url: url,
    data: {

    },
    success: function(js, status){
      console.log("url: " + url + ", status: " + status + ", js: " + JSON.stringify(js));
      if (js.status_code === 60200) {
        var element_select_model = $("#select_model")
        element_select_model.empty();

        for (var i=0; i<js.result.length; i++)
        {
          element_select_model.append("<option>" + js.result[i] + "</option>")
        }
        typeof callback == 'function' && callback()
      } else {
        alert(js.message);
      }
    },
    error: function (resp, status) {
      var js = JSON.parse(resp.responseText)
      console.log("url: " + url + ", status: " + status + ", js: " + JSON.stringify(js));
      alert(js.message);
    }
  })
}


var get_model_info = function (model_name) {
  var url = "basic_intent/get_available_models";

}


var load_model = function (model_name) {

}


$(document).ready(function(){
  get_available_models();

  $("#search").click(function(){

    var model_name = $("#select_model").val();

    if (model_name === "") {
      alert("请选择模型!")
      return null;
    } else {
      get_model_info(model_name);
    }
  });

})

