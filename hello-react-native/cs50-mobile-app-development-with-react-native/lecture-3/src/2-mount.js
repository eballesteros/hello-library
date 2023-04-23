// play with mount cycle
//    constructor(props) -> None
//    render() -> Node
//    componentDidMount() -> None


import React, { Component } from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { Constants } from 'expo';


class Counter extends Component {
  // This is how you would define the constructor
  // constructor() {
  //   super()
  //   this.state = {
  //     count: 0,
  //   }
  // }
  state = {
    count: 0,
  }
  
  // after component finishes mounting
  // great place to init timer
  componentDidMount() {
    // call incrementCount every second
    // setInterval(this.incrementCount, 1000)
    
    // this.timer so we can cleanup when we unmount
    this.timer = setInterval(this.incrementCount, 1000)
  }
  
  // class properties: aromatically bind at creation time
  incrementCount = () => {
    this.setState(prevState => ({count: prevState.count + 1}))
  }
  
  render() {
    return <Text>{this.state.count}</Text>
  }
}

export default class App extends Component {
  render() {
    return (
      <View style={styles.container}>
        <Counter />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: Constants.statusBarHeight,
    backgroundColor: '#ecf0f1',
  },
});

