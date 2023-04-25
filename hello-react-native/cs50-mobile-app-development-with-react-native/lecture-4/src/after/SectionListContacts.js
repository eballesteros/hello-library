import React from 'react'
import {SectionList, Text} from 'react-native'
import PropTypes from 'prop-types'

import Row from './Row'

const renderItem = ({item}) => <Row {...item} />

const renderSectionHeader = ({section}) => <Text>{section.title}</Text>

const SectionListContacts = props => {
  // List[contact] -> Dict[initial, List[contact]]
  const contactsByLetter = props.contacts.reduce((obj, contact) => {
    const firstLetter = contact.name[0].toUpperCase()
    return {
      ...obj,
      [firstLetter]: [...(obj[firstLetter] || []), contact], // overwrite this particular first letter: clone previous + contact;
                                                             // || for the first iteration where it was empty
    }
  }, {})

  const sections = Object.keys(contactsByLetter).sort().map(letter => ({
    data: contactsByLetter[letter],
    title: letter,
  }))

  return <SectionList sections={sections} renderItem={renderItem} renderSectionHeader={renderSectionHeader} />
}

SectionListContacts.propTypes = {
  contacts: PropTypes.array,
}

export default SectionListContacts
