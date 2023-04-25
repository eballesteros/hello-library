import React from 'react';
import { Button, ScrollView, StyleSheet, Text, View, FlatList, SectionList } from 'react-native';
import Constants from 'expo-constants';

import contacts, { compareNames } from './contacts'
import { Row } from './Row'


export default class App extends React.Component {
  // shorthand, creates constructor that only inits state 
  state = {
    showContacts: false,
    contacts: contacts
  }

  toggleContacts = () => {
    this.setState(prevState => ({ showContacts: !prevState.showContacts }))
  }

  sort = () => {
    // sort comes from array prototype
    // we clone the array (contains the reference to the same objects) to force FlatList to rerender
    this.setState(prevState => ({contacts: [...prevState.contacts].sort(compareNames)}))
  }

  // destructuring obj in the inputs
  // renderItem = obj => <Row {...obj.item} />
  renderItem = ({item}) => <Row {...item} />

  renderSectionHeader = obj => <Text>{obj.section.title}</Text>

  render() {
    return (
      <View style={styles.container}>
        <Button title="toggle contacts" onPress={this.toggleContacts} />
        <Button title="sort contacts" onPress={this.sort} />
        {this.state.showContacts && (
          // 1.
          // ScrollView renders all of it's children before displaying, which is not ideal. Takes a few seconds
          // <ScrollView>
          //   {
            //     // React uses the key for easier diffing between its tree and the DOM
            //      contacts.map(contact => <Row key={contact.key} {...contact} />)
            //   }
            // </ScrollView>
            
            // 2.
            // FlatList: --Virtualized-- 
            // - only renders what you see
            // - when something goes out of sight, it is unmounted, so you need to keep it's state outside of the component (for now avoid state)
            // - only updates if props are changed
            //    - we need to use immutability.
            // the FlatList automatically extracts the key if its present
            // <FlatList renderItem={this.renderItem} data={this.state.contacts}>
            // </FlatList> 
            // 3.
            // SectionLust: just like FlatList but with support for sections
            <SectionList 
              renderItem={this.renderItem}
              renderSectionHeader={this.renderSectionHeader}
              sections={[{title: "A", data:this.state.contacts}]}>
            </SectionList> 
          )
        }
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: Constants.statusBarHeight,
  },
});
