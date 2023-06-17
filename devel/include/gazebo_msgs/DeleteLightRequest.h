// Generated by gencpp from file gazebo_msgs/DeleteLightRequest.msg
// DO NOT EDIT!


#ifndef GAZEBO_MSGS_MESSAGE_DELETELIGHTREQUEST_H
#define GAZEBO_MSGS_MESSAGE_DELETELIGHTREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace gazebo_msgs
{
template <class ContainerAllocator>
struct DeleteLightRequest_
{
  typedef DeleteLightRequest_<ContainerAllocator> Type;

  DeleteLightRequest_()
    : light_name()  {
    }
  DeleteLightRequest_(const ContainerAllocator& _alloc)
    : light_name(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _light_name_type;
  _light_name_type light_name;





  typedef boost::shared_ptr< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> const> ConstPtr;

}; // struct DeleteLightRequest_

typedef ::gazebo_msgs::DeleteLightRequest_<std::allocator<void> > DeleteLightRequest;

typedef boost::shared_ptr< ::gazebo_msgs::DeleteLightRequest > DeleteLightRequestPtr;
typedef boost::shared_ptr< ::gazebo_msgs::DeleteLightRequest const> DeleteLightRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace gazebo_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'trajectory_msgs': ['/opt/ros/kinetic/share/trajectory_msgs/cmake/../msg'], 'gazebo_msgs': ['/home/ur3/catkin_SRPP/src/drivers/gazebo_ros_pkgs/gazebo_msgs/msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4fb676dfb4741fc866365702a859441c";
  }

  static const char* value(const ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4fb676dfb4741fc8ULL;
  static const uint64_t static_value2 = 0x66365702a859441cULL;
};

template<class ContainerAllocator>
struct DataType< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "gazebo_msgs/DeleteLightRequest";
  }

  static const char* value(const ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string light_name\n\
";
  }

  static const char* value(const ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.light_name);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct DeleteLightRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::gazebo_msgs::DeleteLightRequest_<ContainerAllocator>& v)
  {
    s << indent << "light_name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.light_name);
  }
};

} // namespace message_operations
} // namespace ros

#endif // GAZEBO_MSGS_MESSAGE_DELETELIGHTREQUEST_H
