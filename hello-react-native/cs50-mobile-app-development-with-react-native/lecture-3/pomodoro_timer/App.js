import { Button, StyleSheet, View } from 'react-native';
import { CountDown } from './CountDown.js'
import { Component } from 'react';
import {vibrate} from './utils'


// This works, but I am not too happy about how I am handling
// toggling/resting the countDowns. Every time I want to do 
// either of those, I modify App.state.countId so the countDown key changes,
// making sure the countDown component is unmounted and a new one is instantiated
class App extends Component {
  state = {
    working: true,
    isRunning: true,
    countId: 0
  }

  toggleCountDowns = () => {
    this.setState(prevState => (
      {
        working: !prevState.working,
        countId: prevState.countId + 1
      }
    ))
  }

  resetCountDown = () => {
    this.setState(prevState => ({countId: prevState.countId + 1}))
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
        <CountDown key={this.state.countId} message={message} startTime={startTime} onFinish={this.onCountdownFinish} isRunning={this.state.isRunning} />
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
});
