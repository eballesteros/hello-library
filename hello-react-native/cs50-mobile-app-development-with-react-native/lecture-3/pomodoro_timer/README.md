# Project 1 - Pomodoro Timer
For this project, you'll be implementing a Pomodoro timer. This timer will help
people trying to use the [Pomodoro Technique](https://en.wikipedia.org/wiki/Pomodoro_Technique).
It will vibrate to tell you when to take breaks or resume working, based on some
determined values. Check out the [staff solution](#staff-solution) for a working
version.


## Requirements
- You may not import libraries other than the below:
  - `expo`
  - `react`
  - `react-native`
  - `prop-types`
- Timer should display minutes and seconds in text
- Timer should count down seconds until it reaches 00:00
- Phone should buzz when timer reaches 0
- Timers should switch between 25 and 5 minutes
- Timer should be able to start, stop, and reset

The aesthetics of the app is up to you!

### Challenge (Not Required)
- Allow the user to input any arbitrary time for the timers (e.g. 5 mins of work time and 5 mins of break)

## Getting Started


- run `npx expo init` to init the project
- `cd` into your project
- `npm install react react-native prop-types` to check all necessary packages are installed
- `npm run ios` to launch (on emulated iPhone)
```

You can now open the app on your phone or simulator by clicking one of the buttons
in the top right. To open on your phone, click the `Share` button and scan the
QR code from the Expo app on your phone. To open in a simulator, click the `Device`
button and select the simulator into which you want to open your app.

When you have the app open in your phone or simulator, try opening [`App.js`](/App.js)
and changing a line. You should see it update on your phone!

You can now begin to work on your app. You may find the vibrate function in
[`/utils`](/utils) helpful. Feel free to import and use it in your app like this:

```javascript
import {vibrate} from './utils'

// causes phone to vibrate
vibrate()
```

Good luck!

## Staff Solution
If you want to play with the staff implementation, you can view it using
Snack at [@jhhayashi/project1-solution](https://snack.expo.io/@jhhayashi/project1-solution).
