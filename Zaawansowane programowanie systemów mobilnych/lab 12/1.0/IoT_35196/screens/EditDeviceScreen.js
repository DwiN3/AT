// EditDeviceScreen.js

import React, { useState, useEffect } from 'react';
import { Text, View, TextInput, TouchableHighlight } from 'react-native';
import { useNavigation, useRoute } from '@react-navigation/native';
import styles from '../styles/NewDeviceStyle';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { devices, devicesList } from '../data/devices';

const EditDeviceScreen = () => {
  const [name, setName] = useState('');
  const [place, setPlace] = useState('');
  const [command, setCommand] = useState('');
  const [color, setColor] = useState('');
  const Colors = ["blue", "yellow", "pink", "red", "green", "purple", "orange"];

  const navigation = useNavigation();
  const route = useRoute();
  const { deviceToEdit } = route.params || {}; 

  useEffect(() => {
    if (deviceToEdit) {
      setName(deviceToEdit.name || '');
      setPlace(deviceToEdit.place || '');
      setCommand(deviceToEdit.command || '');
      setColor(deviceToEdit.color || Colors[0]);
    }
  }, [deviceToEdit]);

  const handleSave = async () => {
    try {
      const updatedDevice = {
        name,
        place,
        command,
        color,
      };
      const storedDevicesList = await AsyncStorage.getItem('devicesList');
      let updatedDevicesList = storedDevicesList ? JSON.parse(storedDevicesList) : [];
  
      if (deviceToEdit) {
        updatedDevicesList = updatedDevicesList.map((device) =>
          device.id === deviceToEdit.id ? { ...device, ...updatedDevice } : device
        );
      } else {
        const newDevice = { ...updatedDevice, id: Date.now().toString() };
        updatedDevicesList.push(newDevice);
      }
      await AsyncStorage.setItem('devicesList', JSON.stringify(updatedDevicesList));
      console.log('Edycja udana');
      navigation.navigate('Devices');
    } catch (error) {
      console.error('Error saving device:', error);
    }
  };

  const exit = () => {
    navigation.navigate('Devices');
  };

  const handleColorSelect = (selectedColor) => {
    setColor(selectedColor);
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Name"
        value={name}
        onChangeText={(text) => setName(text)}
      />
      <TextInput
        style={styles.input}
        placeholder="Place"
        value={place}
        onChangeText={(text) => setPlace(text)}
      />
      <TextInput
        style={styles.input}
        placeholder="Command"
        value={command}
        onChangeText={(text) => setCommand(text)}
      />

      <View><Text style={{ fontSize:20, marginVertical:12 }} >Colors</Text></View>
      <View style={styles.colorsContainer}>
        {Colors.map((c) => (
          <TouchableHighlight
            key={c}
            style={[
              styles.circleOneColor,
              { backgroundColor: c, borderWidth: color === c ? 2 : 0 },
            ]}
            onPress={() => handleColorSelect(c)}
          >
            <View style={{ width: 40, height: 40, borderRadius: 20 }} />
          </TouchableHighlight>
        ))}
      </View>

      <TouchableHighlight
        style={[styles.buttonContainer, styles.button, { backgroundColor: 'green' }]}
        onPress={handleSave}
      >
        <Text style={styles.buttonText}>Save</Text>
      </TouchableHighlight>

      <TouchableHighlight
        style={[styles.buttonContainer, styles.button, { backgroundColor: 'red' }]}
        onPress={exit}
      >
        <Text style={styles.buttonText}>Cancel</Text>
      </TouchableHighlight>
    </View>
  );
};

export default EditDeviceScreen;