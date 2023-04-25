import React from 'react';
import { Button, ScrollView, StyleSheet, Text, View } from 'react-native';
// import Constants from 'expo-constants';

import contacts from './contacts'

export default class App extends React.Component {
  // shorthand, creates constructor that only inits state 
  state = {
    showContacts: false,
  }

  toggleContacts = () => {
    this.setState(prevState => ({showContacts: !prevState.showContacts}))
  }

  render() {
    return (
      <View style={styles.container}>
        <Text>"Hi mom!"</Text>
        <Button title="toggle contacts" onPress={this.toggleContacts} />
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    // paddingTop: Constants.statusBarHeight,
  },
});
