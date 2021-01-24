$(() => {
  loadCameraOnConnect({
    host: "odroid-black.local",
    container: '#cam-frame',
    port: 1190,
    image_url: '/stream.mjpg',
    data_url: '/settings.json',
    attrs: {
      width: 0,
      height: 0
    }
  });

  attachRobotConnectionIndicator("#status");

  function setupLayoutHandlers() {
    $("#vrt-layout-btn").click(() => {
      $("#content-area").removeClass("vert-split hor-split cam-fullscreen");
      $("#content-area").addClass("vert-split");
    });
    $("#hor-layout-btn").click(() => {
      $("#content-area").removeClass("vert-split hor-split cam-fullscreen");
      $("#content-area").addClass("hor-split");
    });
    $("#fs-layout-btn").click(() => {
      $("#content-area").removeClass("vert-split hor-split cam-fullscreen");
      $("#content-area").addClass("cam-fullscreen");
    });
  }
  setupLayoutHandlers();

  function createvalue(parent, tar, startclosed, isleaf) {
    var tag = $(`<div>${tar}</div>`);
    var contents = $(`<div></div>`);

    var wrapper = $(`<div></div>`);
    if (isleaf) {
      tag.addClass("leaf-tag");
      contents.addClass("leaf-itm");
      tag.addClass("optn-open");
    } else {
      tag.attr("tabindex", "0");
      tag.addClass("stem-tag");
      contents.addClass("stem-itm");

      const clickHandler = (() => {
        if (contents.is(":hidden")) {
          tag.addClass("optn-open");
        } else {
          tag.removeClass("optn-open");
        }
        contents.slideToggle();
      });
      tag.click(clickHandler);
      tag.keypress((e) => {
        if (e.key === ' ' || e.key === 'Spacebar' || event.key === "Enter") {
          clickHandler();
        }
      });

      if (startclosed) {
        contents.hide();
      } else {
        tag.addClass("optn-open")
      }
    }
    wrapper.append(tag);
    wrapper.append(contents);
    parent.append(wrapper);

    return (contents);
  }

  function getType(obj) {
    const primitiveType = typeof obj;
    if (primitiveType == "string" || primitiveType == "boolean" || primitiveType == "number") {
      return (primitiveType);
    } else if (Array.isArray(obj)) {
      if (!obj.length) return ("unknown array")
      const arrayType = typeof obj[0];
      if (arrayType == "string" || arrayType == "boolean" || arrayType == "number") {
        return (arrayType + " array");
      }
    }
    console.log(obj)
    throw new TypeError(obj + " (" + primitiveType + ") is not a valid type for networktables");
  }

  function toType(str, type) { //returns null if invalid
    function toTypeSingle(str, type) {
      switch (type) {
        case "string":
          return (str);
          break;
        case "boolean":
          const processed = str.trim().toLowerCase();
          if (processed == "true") return (true);
          else if (processed == "false") return (false);
          else return (null);
          break;
        case "number":
          const parsed = parseFloat(str);
          if (Number.isNaN(parsed)) return (null);
          return (parsed);
          break;
        default:
          return (null);
      }
    }
    if (type.endsWith("array")) {
      const arrayType = type.slice(0, -6);
      const splt = str.split(",");
      var res = [];
      for (var i = 0; i < splt.length; i++) {
        const proc = toTypeSingle(splt[i], arrayType);
        if (proc == null) return (null);
        res.push(proc);
      }
      return (res);
    } else {
      const primitiveType = type;
      return (toTypeSingle(str, type));
    }
  }

  function createInputElem(value, type, path) {
    const inp = $(`<input>`);
    inp.val(value);
    inp.change(() => {
      const proc = toType(inp.val(), type);
      if (proc === null) {
        inp.val(NetworkTables.getValue(path));
      } else {
        NetworkTables.putValue(path, proc);
      }
    });

    NetworkTables.addKeyListener(path, function(key, value, isNew) {
      if (isNew) console.log("Unexpected: new value when not new");
      inp.val(value);
    });
    return (inp);
  }

  function putKey(path, value, startclosed, fullpath, optRoot) {
    const root = optRoot || $("#nt-list");

    if (!path.length) {
      if (root.children().length) {
        root.children().first().val(value);
        return;
      }
      const type = getType(value);
      const inpElem = createInputElem(value, type, fullpath);
      root.append(inpElem);
      root.append($(`<div class=leaf-itm-info>(${type})</div>`))
      return;
    }

    const rest = path.slice(1);
    const tar = path[0];
    let found = false;
    root.children().each((indx, val) => {
      let jval = $(val);
      if (jval.children().first().text() == tar) {
        window.setTimeout(() => {
          putKey(rest, value, true, fullpath, jval.children().eq(1))
        }, 10); //allow time for rendering etc.
        found = true;
      }
    })
    if (found) return found;
    putKey(rest, value, true, fullpath, createvalue(root, tar, startclosed, !rest.length));
  }

  function splitpath(str) {
    const split = str.split("/");
    return split[0] ? split : split.slice(1);
  }

  function addCameraOptions() {
    const prePath = "/SmartDashboard/vision/active_mode/";
    const camOpts = NetworkTables.getValue(prePath + "options");
    $("#cam-selection").empty();
    if (camOpts) {
      for (var i = 0; i < camOpts.length; i++) {
        const thisOpt = camOpts[i];
        const radioId = "camOptId" + NetworkTables.keyToId(thisOpt);
        let opt = $(`<input type="radio" id="${radioId}" name="camOpts">`);
        let lab = $(`<label for="${radioId}">${thisOpt}</label>`)
        $("#cam-selection").append(opt);
        $("#cam-selection").append(lab);
        $("#cam-selection").append($("<br>"));
        opt.change(() => {
          if (opt.is(':checked')) {
            NetworkTables.putValue(prePath + "selected", thisOpt);
          }
        });
      }
      const currSel = $("#camOptId" + NetworkTables.getValue(prePath + "selected"));
      currSel.prop('checked', true);
      NetworkTables.addKeyListener(prePath + "selected", (key, value) => {
        $(`#camOptId${NetworkTables.keyToId(value)}`).prop('checked', true);
      });
    } else {
      $("#cam-selection").append("Unable to get active_mode options (check path?)");
    }
  }
  NetworkTables.addRobotConnectionListener(() => {
    window.setTimeout(() => {
      $("#nt-list").empty();
      const keys = NetworkTables.getKeys();
      keys.forEach((key) => {
        putKey(splitpath(key), NetworkTables.getValue(key), false, key);
      });
      addCameraOptions();
    }, 100);
  });
  NetworkTables.addGlobalListener((key, value, isNew) => {
    if (!isNew) return;
    putKey(splitpath(key), NetworkTables.getValue(key), false, value);
  });
})
