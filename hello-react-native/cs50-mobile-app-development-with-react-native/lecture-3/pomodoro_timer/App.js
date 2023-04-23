import { Button, StyleSheet, Text, View } from 'react-native';
import { CountDown } from './CountDown.js'
import React, { Component } from 'react';
import {vibrate} from './utils'


// this version uses React.createRef to enable
// resetCountDown to call CountDown.resetState
class App extends Component {
  state = {
    working: true,
    isRunning: true,
  }

  countDownRef = React.createRef()

  resetCountDown = () => {
    this.countDownRef.current.resetState()
  }

  toggleCountDowns = () => {
    this.setState(prevState => (
      {
        working: !prevState.working,
      }
    ))
    // No need since the above change causes a startTime change which causes
    // CountDown.componentDidUpdate to reset the timer
    // this.resetCountDown()
  }

  onCountdownFinish = () => {
    this.toggleCountDowns()
    vibrate()
  }

  toggleRunningState = () => {
    this.setState(prevState => ({isRunning: !prevState.isRunning}))
  }

  render() {
    const message = this.state.working ? 'Working' : 'Resting';
    const startTime = this.state.working ? 3: 2
    
    return (
      <View style={styles.container}>
        <Text style={styles.text}> {message} </Text>
        <CountDown ref={this.countDownRef} startTime={startTime} onFinish={this.onCountdownFinish} isRunning={this.state.isRunning} />
        <View>
          <Button onPress={this.toggleRunningState} title='start/stop' />
          <Button onPress={this.resetCountDown} title='reset' />
        </View>
      </View>
    );
  }
}

export default App


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  // I don't love the fact that I have 2 different styles
  // (one here and one in CountDown) for text.
  // But it doesn't bother me enough to fix it. 
  text: {fontSize: 50}
});
