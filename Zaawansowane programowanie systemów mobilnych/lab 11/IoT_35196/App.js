// Apps.js

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, TouchableOpacity } from 'react-native'; 
import DevicesScreen from './screens/DevicesScreen';
import NewDeviceScreen from './screens/NewDeviceScreen';
import ConnectionScreen from './screens/ConnectionScreen';
import EditDeviceScreen from './screens/EditDeviceScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

const BottomTabNavigator = () => {
  return (
    <Tab.Navigator 
      screenOptions={{
        headerTitleAlign: 'center',
      }}
      tabBar={({ state, descriptors, navigation }) => (
        <View style={{ flexDirection: 'row', height: 60, backgroundColor: 'lightblue', alignItems: 'center', justifyContent: 'space-around' }}>
          {state.routes.map((route, index) => {
            const { options } = descriptors[route.key];
            const label = options.tabBarLabel !== undefined ? options.tabBarLabel : options.title !== undefined ? options.title : route.name;
            
            const isFocused = state.index === index;
            const textStyle = {
              fontSize: isFocused ? 18 : 16,
              color: isFocused ? 'blue' : 'black',
            };

            const onPress = () => {
              const event = navigation.emit({
                type: 'tabPress',
                target: route.key,
              });

              if (!isFocused && !event.defaultPrevented) {
                navigation.navigate(route.name);
              }
            };

            return (
              <TouchableOpacity
                key={index}
                accessibilityRole="button"
                accessibilityState={isFocused ? { selected: true } : {}}
                accessibilityLabel={options.tabBarAccessibilityLabel}
                testID={options.tabBarTestID}
                onPress={onPress}
                style={{ flex: 1, alignItems: 'center' }}
              >
                <Text style={textStyle}>{label}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      )}
    >
      <Tab.Screen
        name="Devices"
        component={DevicesScreen}
        options={{
          tabBarIcon: ({ focused, color, size }) => null, 
        }}
      />
      <Tab.Screen
        name="Connection"
        component={ConnectionScreen}
        options={{
          tabBarIcon: ({ focused, color, size }) => null, 
        }}
      />
    </Tab.Navigator>
  );
};

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen 
          name="Device" 
          component={BottomTabNavigator} 
          options={{ 
            headerShown: false,
          }} 
        />
        <Stack.Screen 
          name="New Device" 
          component={NewDeviceScreen} 
          options={{
            headerTitleStyle: { fontWeight: 'bold' },
          }}
        />
        <Stack.Screen 
          name="Connection" 
          component={ConnectionScreen} 
          options={{
            headerTitleStyle: { fontWeight: 'bold' },
          }}
        />
        <Stack.Screen 
          name="Edit Device" 
          component={EditDeviceScreen} 
          options={{
            headerTitleStyle: { fontWeight: 'bold' },
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;