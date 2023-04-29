import React from 'react'
import {Button, KeyboardAvoidingView, StyleSheet, TextInput, View} from 'react-native'
import {Constants} from 'expo'

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    paddingTop: Constants.statusBarHeight,
  },
  input: {
    borderWidth: 1,
    borderColor: 'black',
    minWidth: 100,
    marginTop: 20,
    marginHorizontal: 20,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 3,
  },
})

export default class AddContactForm extends React.Component {
  state = {
    name: '',
    phone: '',
    isFormValid: false,
  }


  // we could also be passing a callback (gets invoked after the state is updated)
  // in handlePhoneChange and handleNameChange (ex: this.setState({phone}, this.validateForm)
  componentDidUpdate(prevProps, prevState) {
    // validateForm also updates the state, so without this check we would stuck in an 
    // infinite update loop
    if (this.state.name !== prevState.name || this.state.phone !== prevState.phone) {
      this.validateForm()
    }
  }

  // this is a pattern to avoid needing 1 handler per field in the form
  // shorthand notation (instead of 1 line return).
  getHandler = key => val => {
    // [key] means evaluate this expression, cast to str, that will be the key
    this.setState({[key]: val})
  }

  // same as above without the shorthand notation
  // getHandler = key => {
  //   return val => {
  //     this.setState({[key]: val})
  //   }
  // }

  // you don't really need to declare them here. You can pass them in the "form" directly
  // but on the other hand you will get one call per field in the form for every render
  // handleNameChange = this.getHandler('name') // val => { this.setState({name: val}) }
  // handlePhoneChange = this.getHandler('phone')

    /*
  handleNameChange = name => {
    this.setState({name})
  }
  */

  handlePhoneChange = phone => {
    // + tries to cast to a number (returns NaN if not, except "" where it returns 0)
    // NaN is not >= 0
    if (+phone >= 0 && phone.length <= 10) {
      this.setState({phone})
    }
  }

  validateForm = () => {
    console.log(this.state)
    const names = this.state.name.split(' ')
    // about names[0] names[1]: 
    // A trailing (or preceding) whitespace appears as a word when you do split(' '). This catches that 
    if (+this.state.phone >= 0 && this.state.phone.length === 10 && names.length >= 2 && names[0] && names[1]) {
      this.setState({isFormValid: true})
    } else {
      this.setState({isFormValid: false})
    }
  }

  // validateForm2 = () => {
  //   if (+this.state.phone >= 0 && this.state.phone.length === 10 && this.state.name.length >= 3) {
  //     return true
  //   } else {
  //     return false
  //   }
  // }

  // we could check that the name and phone are valid here, but disabling is a better UI
  handleSubmit = () => {
    // we pass the state UP through the prompt
    this.props.onSubmit(this.state)
  }

  render() {
    return (
      // KeyboardAvoidingView is good for "short" forms. It moves the view out of the 
      // way of the keyboard. You can configure behavior (check docs)
      <KeyboardAvoidingView behavior="padding" style={styles.container}>
        <TextInput
          style={styles.input}
          value={this.state.name}
          onChangeText={this.getHandler('name')}
          placeholder="Name"
        />
        <TextInput
          keyboardType="numeric"
          style={styles.input}
          value={this.state.phone}
          onChangeText={this.getHandler('phone')}
          placeholder="Phone"
        />
        
        <Button title="Submit" 
                onPress={this.handleSubmit} 
                // you could call this.validateForm2() here instead, but you want to avoid computation in
                // the render step, and leave it for other steps of the update cycle (so the component renders fast)
                disabled={!this.state.isFormValid} />
      </KeyboardAvoidingView>
    )
  }
}
