
// warnings: only on dev mode
console.warn("this is a less aggressive warning")

// errors
console.error("Error message")
throw Error

// ## Chrome dev tools (debugger)
// - remember react-native runs 2 thread (UI, JS)
// - They communicate asynchronously though a bridge
// - you can run the JS in a different device! (like the browser)
// in the simulator, DEVICE -> SHAKE -> OPEN JS DEBUGGER

// How do you troubleshoot layout bugs?
// in the simulator, DEVICE -> SHAKE -> SHOW ELEMENT INSPECTOR
//  It doesn't allow you to live-update these elements, for that you can use react-devtools

// ## react-devtools
// 
// ### Install
// `npm install -g react-devtools` (-g stands for global)
// 
// ### Run
// `react-devtools`